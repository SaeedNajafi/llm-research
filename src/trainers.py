"""The main module for different objectives to train the policy (llm)."""

from itertools import chain
from typing import Any, List, Optional

import torch
from absl import flags, logging
from torch.utils.data import DataLoader

from src.llm import LLM, LLMGenerationOutput
from src.metrics import RewardCalculator
from src.utils.dpo_utils import preference_loss
from src.utils.general_utils import DictDataset
from src.utils.rl_utils import compute_entropy_loss, form_returns, normalize_signals

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "rl_sample_size", 8, "The number of samples to generate from the policy used for both on/off-policy learnings."
)
flags.DEFINE_integer(
    "iterative_chunk_size", 2, "The number of return sequences to generate from LLM at the same time in parallel."
)
flags.DEFINE_boolean("with_baseline", False, "Whether to use the baseline reward in RL objective.")
flags.DEFINE_string("reward_normalization_type", "zscore", "zscore | mml_normalize | normalize | rloo_normalize | no_normalize")
flags.DEFINE_float("baseline_momentum", 0.1, "momentum used to compute the average reward in the RL baseline.")
flags.DEFINE_boolean("compute_per_step_entropy", False, "Whether to add per-step entropy to the loss in RL training.")
flags.DEFINE_float("entropy_coef", 0.1, "Coefficient used to mix per-step entropy loss in RL training.")
flags.DEFINE_string("objective_type", "reinforce", "Different objectives to get the loss for training the llm.")
flags.DEFINE_string("mml_version", "version_1", "Which version to compute the mml loss.")
flags.DEFINE_boolean(
    "include_policy_ref_kl", False, "Whether to apply the KL divergence between the policy and the reference policy."
)
flags.DEFINE_float(
    "policy_ref_kl_coef", 0.1, "Coefficient to apply the KL divergence between the policy and the reference policy."
)
flags.DEFINE_boolean("compute_true_validation_loss", True, "True or False?")
flags.DEFINE_boolean("preference_optimization_with_reference", False, "True or False? reference free DPO/IPO or not?")
flags.DEFINE_float("preference_beta", 0.1, "beta in DPO/IPO loss.")
flags.DEFINE_boolean("preference_ipo", False, "Whether to use the IPO version of preference optimization.")
flags.DEFINE_float("preference_label_smoothing", 0.1, "label smoothing used for conservative DPO/IPO.")


class LossCalculator:

    def __init__(
        self,
        policy_lm: LLM,
        objective_type: str,
        reward_name: str,
        weights_base_folder: str,
        value_lm: Optional[LLM] = None,
        ref_policy_lm: Optional[LLM] = None,
    ):
        super().__init__()
        self.policy_lm = policy_lm
        if value_lm is not None:
            self.value_lm = value_lm
        if ref_policy_lm is not None:
            self.ref_policy_lm = ref_policy_lm
        if FLAGS.with_baseline:
            # For simple, time-step based average return.
            self.baseline_returns = torch.zeros(
                (FLAGS.input_max_length + FLAGS.output_max_length + 1,), dtype=torch.float64, device=self.policy_lm.device
            )

        self.objective_type = objective_type
        self.reward_calculator = RewardCalculator(reward_name, weights_base_folder)

    def teacher_forcing_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        log_likelihood = self.policy_lm.train(batch, per_step_scores=False, to_train=True)
        loss = -torch.mean(log_likelihood, dim=0)
        return loss

    def compute_true_validation_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Compute the true validation loss under the model."""
        if FLAGS.compute_true_validation_loss:
            # It uses teacher forcing over the validation data to compute the loss.
            validation_log_likelihood = self.policy_lm.train(batch, per_step_scores=False, to_train=False)
            validation_true_loss = -torch.mean(validation_log_likelihood, dim=0).detach().float()
            return validation_true_loss
        else:
            raise ValueError("--compute_true_validation_loss should be true!")

    def sample_and_generate_details(self, batch: torch.utils.data.Dataset, to_train: bool = True) -> Any:
        """Compute per-step information while sampling.

        We can also give prior labels which will compute the logits of
        it. Useful for computing the log-p of reference policy for the
        samples.
        """
        assert FLAGS.rl_sample_size > 1
        num_iterative_calls = FLAGS.rl_sample_size // FLAGS.iterative_chunk_size
        partial_generations = []
        final_log_ps = []
        actual_lens = []
        labels_to_consider = []
        token_final_log_ps = []
        full_generated_sample_ids = []
        logits = []
        for call_idx in range(num_iterative_calls):
            llm_generation_outputs: List[LLMGenerationOutput] = self.policy_lm.generation_pass(
                batch,
                top_p=FLAGS.train_top_p,
                temperature=FLAGS.train_temperature,
                num_return_sequences=FLAGS.iterative_chunk_size,
                to_train=to_train,
                use_cache=True,
                per_step_scores=True,
                iterative_rl_sampling=False,
                generate_partial_sequences=True,
            )
            final_log_ps_per_call = llm_generation_outputs[0].final_log_ps
            actual_lens_per_call = llm_generation_outputs[0].actual_lens
            labels_to_consider_per_call = llm_generation_outputs[0].labels_to_consider
            token_final_log_ps_per_call = llm_generation_outputs[0].token_final_log_ps
            full_generated_sample_ids_per_call = llm_generation_outputs[0].full_generated_sample_ids
            partial_generations_per_call = llm_generation_outputs[0].partially_generated_sequences
            logits_per_call = llm_generation_outputs[0].logits
            batch_size_extended, seq_len, vocab_size = logits_per_call.size()
            batch_partial_generations_per_call = []
            batch_size = batch_size_extended // FLAGS.iterative_chunk_size
            for b_idx in range(batch_size):
                batch_partial_generations_per_call.append(
                    [
                        partial_generations_per_call[
                            b_idx * FLAGS.iterative_chunk_size : (b_idx + 1) * FLAGS.iterative_chunk_size
                        ]
                    ]
                )
            final_log_ps.append(final_log_ps_per_call.view(batch_size, FLAGS.iterative_chunk_size))
            actual_lens.append(actual_lens_per_call.view(batch_size, FLAGS.iterative_chunk_size))
            labels_to_consider.append(labels_to_consider_per_call.view(batch_size, FLAGS.iterative_chunk_size, -1))
            token_final_log_ps.append(token_final_log_ps_per_call.view(batch_size, FLAGS.iterative_chunk_size, -1))
            full_generated_sample_ids.append(
                full_generated_sample_ids_per_call.view(batch_size, FLAGS.iterative_chunk_size, -1)
            )
            logits.append(logits_per_call.view(batch_size, FLAGS.iterative_chunk_size, seq_len, vocab_size))
            partial_generations.append(batch_partial_generations_per_call)
        sequence_log_probs = torch.cat(final_log_ps, dim=1)
        actual_lens_tensor = torch.cat(actual_lens, dim=1)
        batch_size = sequence_log_probs.size()[0]
        final_labels_to_consider = []
        token_log_ps = []
        full_sample_ids = []
        final_logits = []
        partial_samples = []
        for b_idx in range(batch_size):
            labels_to_consider_arr = []
            token_log_ps_arr = []
            full_sample_ids_arr = []
            logits_arr = []
            partial_samples_arr = []
            for call_idx in range(num_iterative_calls):
                for chunk_idx in range(FLAGS.iterative_chunk_size):
                    labels_to_consider_arr.append(labels_to_consider[call_idx][b_idx, chunk_idx, :])
                    token_log_ps_arr.append(token_final_log_ps[call_idx][b_idx, chunk_idx, :])
                    full_sample_ids_arr.append(full_generated_sample_ids[call_idx][b_idx, chunk_idx, :])
                    logits_arr.append(logits[call_idx][b_idx, chunk_idx, :, :])
                    partial_samples_arr.append(partial_generations[call_idx][b_idx][0][chunk_idx])

            final_labels_to_consider.append(labels_to_consider_arr)
            token_log_ps.append(token_log_ps_arr)
            full_sample_ids.append(full_sample_ids_arr)
            final_logits.append(logits_arr)
            partial_samples.append(partial_samples_arr)

        max_len = actual_lens_tensor.max()
        # Pad sequences up until max_len
        logits_flattened = []
        token_log_ps_flattened = []
        full_sample_ids_flattened = []
        flattened_labels_to_consider = []
        for b_idx in range(batch_size):
            for sample_idx in range(FLAGS.rl_sample_size):
                # pad labels to consider
                cur_labels_tensor = final_labels_to_consider[b_idx][sample_idx]
                cur_len = cur_labels_tensor.size()[0]
                pad_label_tensor = torch.tensor(
                    [-100] * (max_len - cur_len), dtype=cur_labels_tensor.dtype, device=cur_labels_tensor.device
                )
                flattened_labels_to_consider.append(torch.cat((cur_labels_tensor, pad_label_tensor), dim=0))

                # Pad samples with the last one, corresponding to padding the generated ids with the pad token.
                last_partial_sample = partial_samples[b_idx][sample_idx][-1]
                partial_samples[b_idx][sample_idx].extend([last_partial_sample] * (max_len - cur_len))

                # pad token log_ps
                cur_token_log_ps_tensor = token_log_ps[b_idx][sample_idx]
                pad_token_log_ps_tensor = torch.tensor(
                    [0.0] * (max_len - cur_len), dtype=cur_token_log_ps_tensor.dtype, device=cur_token_log_ps_tensor.device
                )
                token_log_ps_flattened.append(torch.cat((cur_token_log_ps_tensor, pad_token_log_ps_tensor), dim=0))

                # pad logits
                cur_logits = final_logits[b_idx][sample_idx]
                cur_len, vocab_size = cur_logits.size()
                pad_logits = torch.zeros((max_len - cur_len, vocab_size), dtype=cur_logits.dtype, device=cur_logits.device)
                logits_flattened.append(torch.cat((cur_logits, pad_logits), dim=0))

                # pad the sample ids.
                cur_sample_ids = full_sample_ids[b_idx][sample_idx]
                cur_len = cur_sample_ids.size()[0]
                pad_sample_ids = torch.tensor(
                    [self.policy_lm.tokenizer.pad_token_id] * (max_len - cur_len),
                    dtype=cur_sample_ids.dtype,
                    device=cur_sample_ids.device,
                )
                full_sample_ids_flattened.append(torch.cat((cur_sample_ids, pad_sample_ids), dim=0))

        final_logits = torch.stack(logits_flattened, dim=0).view(batch_size, FLAGS.rl_sample_size, -1, vocab_size)
        final_token_log_ps = torch.stack(token_log_ps_flattened, dim=0).view(batch_size, FLAGS.rl_sample_size, -1)
        final_full_sample_ids = torch.stack(full_sample_ids_flattened, dim=0).view(batch_size * FLAGS.rl_sample_size, -1)
        final_labels_to_consider = torch.stack(flattened_labels_to_consider, dim=0).view(batch_size, FLAGS.rl_sample_size, -1)

        return_data = {
            "sequence_log_probs": sequence_log_probs,
            "actual_lens": actual_lens_tensor,
            "partial_samples": partial_samples,
            "labels_to_consider": final_labels_to_consider,
            "token_log_ps": final_token_log_ps,
            "logits": final_logits,
            "final_full_sample_ids": final_full_sample_ids,
        }

        # We should add a padding code to here!
        return return_data

    def find_reference_scores(
        self,
        batch: torch.utils.data.Dataset,
        teacher_forcing_labels: torch.LongTensor,
    ) -> Any:
        """Compute token log likelihoods for the previous samples using a reference model."""
        # Compute log likelihood of the reference model for the samples.
        num_samples_per_example = FLAGS.rl_sample_size
        input_texts = batch["texts"]
        batch_size = len(input_texts)
        row_ids = batch["row_ids"]
        expanded_input_texts = list(chain.from_iterable([[text] * num_samples_per_example for text in input_texts]))
        expanded_row_ids = list(chain.from_iterable([[row_id] * num_samples_per_example for row_id in row_ids]))
        data = self.ref_policy_lm.prepare_text_for_inference(texts=expanded_input_texts, row_ids=expanded_row_ids)
        dataset = DictDataset(data)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=len(dataset),
            num_workers=1,
        )
        teacher_forcing_labels = teacher_forcing_labels.to(self.ref_policy_lm.device)
        for ref_batch in dataloader:
            # This only happens once.
            llm_generation_outputs: List[LLMGenerationOutput] = self.ref_policy_lm.generation_pass(
                ref_batch,
                top_p=FLAGS.test_top_p,
                temperature=FLAGS.test_temperature,
                num_return_sequences=1,
                to_train=False,
                use_cache=True,
                per_step_scores=True,
                iterative_rl_sampling=False,
                generate_partial_sequences=False,
                teacher_forcing_labels=teacher_forcing_labels,
            )
            token_final_log_ps = llm_generation_outputs[0].token_final_log_ps.view(batch_size, num_samples_per_example, -1)
            sequence_log_ps = llm_generation_outputs[0].final_log_ps.view(batch_size, num_samples_per_example)

        return sequence_log_ps, token_final_log_ps

    def reinforce_loss(self, batch: torch.utils.data.Dataset, terminal_reward_only: bool = False) -> torch.Tensor:
        """Use reinforce with per-step to compute the loss."""
        sample_data = self.sample_and_generate_details(batch)
        gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
        per_step_rewards = self.reward_calculator.compute_per_step_rewards(
            gold_answers,
            sample_data["partial_samples"],
            output_template=self.policy_lm.output_template,
            templated_rewards=True,
            terminal_reward_only=terminal_reward_only,
        )
        per_step_rewards = torch.tensor(per_step_rewards, dtype=torch.float64, device=self.policy_lm.device)
        masks = sample_data["labels_to_consider"] != -100
        if FLAGS.include_policy_ref_kl:
            _, ref_token_log_ps = self.find_reference_scores(batch, sample_data["final_full_sample_ids"])
            if terminal_reward_only:
                per_step_rewards += FLAGS.policy_ref_kl_coef * torch.sum(
                    ref_token_log_ps.to(self.policy_lm.device), dim=2, keepdim=True
                )
            else:
                per_step_rewards += FLAGS.policy_ref_kl_coef * ref_token_log_ps.to(self.policy_lm.device)

        if terminal_reward_only:
            returns = per_step_rewards
            normalized_returns = normalize_signals(returns, normalization_type=FLAGS.reward_normalization_type)
        else:
            returns = form_returns(per_step_rewards * torch.where(masks, 1, 0))
            normalized_returns = normalize_signals(returns, masks=masks, normalization_type=FLAGS.reward_normalization_type)

        if FLAGS.with_baseline:
            max_len = normalized_returns.size()[2]
            current_minibatch_baseline_returns = torch.sum(torch.sum(normalized_returns, dim=0), dim=0)
            non_zero_counts = torch.sum(torch.sum(torch.where(masks, 1, 0), dim=0), dim=0)
            current_minibatch_baseline_returns = current_minibatch_baseline_returns / non_zero_counts[:max_len]
            old_average = self.baseline_returns[:max_len]
            normalized_returns_after_baselines = normalized_returns - old_average
            if terminal_reward_only:
                seq_log_ps = torch.sum(sample_data["token_log_ps"] * torch.where(masks, 1, 0), dim=2)
                loss = -torch.mean(
                    torch.mean(
                        seq_log_ps * normalized_returns_after_baselines.squeeze(-1),
                        dim=1,
                    ),
                    dim=0,
                )
            else:
                loss = -torch.mean(
                    torch.mean(torch.sum(sample_data["token_log_ps"] * normalized_returns_after_baselines, dim=2), dim=1), dim=0
                )

            # Update the baseline
            alpha = FLAGS.baseline_momentum
            self.baseline_returns[:max_len] = alpha * old_average + (1.0 - alpha) * current_minibatch_baseline_returns

            msg = f"\nbaseline reward used: {self.baseline_returns.tolist()}"
            logging.info(msg)

        else:
            if terminal_reward_only:
                seq_log_ps = torch.sum(sample_data["token_log_ps"] * torch.where(masks, 1, 0), dim=2)
                loss = -torch.mean(torch.mean(seq_log_ps * normalized_returns.squeeze(-1), dim=1), dim=0)
            else:
                loss = -torch.mean(torch.mean(torch.sum(sample_data["token_log_ps"] * normalized_returns, dim=2), dim=1), dim=0)

        # Compute the per-step entropy if requested.
        if FLAGS.compute_per_step_entropy:
            entropy_loss = compute_entropy_loss(
                sample_data["labels_to_consider"],
                sample_data["actual_lens"],
                sample_data["token_log_ps"],
                sample_data["logits"],
                approximate_entropy=False,
                perform_length_normalization=True,
            )
            msg = f"\nentropy loss computed: {entropy_loss}"
            logging.info(msg)
            coefficient = FLAGS.entropy_coef
            if FLAGS.include_policy_ref_kl:
                # necessary to train with kl penalty between ref and policy.
                coefficient += FLAGS.policy_ref_kl_coef
            loss += coefficient * entropy_loss

        return loss

    def preference_optimization_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Use preference optimization to compute the loss."""
        assert FLAGS.rl_sample_size > 1
        num_iterative_calls = FLAGS.rl_sample_size // FLAGS.iterative_chunk_size
        generations = []
        final_log_ps = []
        full_generated_sample_ids = []
        for call_idx in range(num_iterative_calls):
            llm_generation_outputs: List[LLMGenerationOutput] = self.policy_lm.generation_pass(
                batch,
                top_p=FLAGS.train_top_p,
                temperature=FLAGS.train_temperature,
                num_return_sequences=FLAGS.iterative_chunk_size,
                to_train=True,
                use_cache=True,
                per_step_scores=False,
                iterative_rl_sampling=False,
            )
            generations_per_call = llm_generation_outputs[0].predictions_str
            final_log_ps_per_call = llm_generation_outputs[0].final_log_ps
            full_generated_sample_ids_per_call = llm_generation_outputs[0].full_generated_sample_ids
            batch_generations_per_call = []
            batch_size = len(generations_per_call) // FLAGS.iterative_chunk_size
            for b_idx in range(batch_size):
                batch_generations_per_call.append(
                    [generations_per_call[b_idx * FLAGS.iterative_chunk_size : (b_idx + 1) * FLAGS.iterative_chunk_size]]
                )
            final_log_ps.append(final_log_ps_per_call.view(batch_size, FLAGS.iterative_chunk_size))
            generations.append(batch_generations_per_call)
            full_generated_sample_ids.append(
                full_generated_sample_ids_per_call.view(batch_size, FLAGS.iterative_chunk_size, -1)
            )

        sequence_log_probs = torch.cat(final_log_ps, dim=1)
        batch_size = sequence_log_probs.size()[0]
        # Compute the rewards.
        gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
        samples = []
        full_sample_ids = []
        for b_idx in range(batch_size):
            sample_arr = []
            full_sample_ids_arr = []
            for call_idx in range(num_iterative_calls):
                samples_strings = generations[call_idx][b_idx][0]
                sample_arr.extend(samples_strings)
                for chunk_idx in range(FLAGS.iterative_chunk_size):
                    full_sample_ids_arr.append(full_generated_sample_ids[call_idx][b_idx, chunk_idx, :])

            # Add extra third dimension.
            sample_arr_extended = [[each] for each in sample_arr]
            samples.append(sample_arr_extended)
            full_sample_ids.append(full_sample_ids_arr)

        full_sample_ids_flattened = []
        # Find max len for padding.
        max_len = 0
        for b_idx in range(batch_size):
            for sample_idx in range(FLAGS.rl_sample_size):
                # pad the sample ids.
                cur_sample_ids = full_sample_ids[b_idx][sample_idx]
                cur_len = cur_sample_ids.size()[0]
                if cur_len > max_len:
                    max_len = cur_len

        for b_idx in range(batch_size):
            for sample_idx in range(FLAGS.rl_sample_size):
                # pad the sample ids.
                cur_sample_ids = full_sample_ids[b_idx][sample_idx]
                cur_len = cur_sample_ids.size()[0]
                pad_sample_ids = torch.tensor(
                    [self.policy_lm.tokenizer.pad_token_id] * (max_len - cur_len),
                    dtype=cur_sample_ids.dtype,
                    device=cur_sample_ids.device,
                )
                full_sample_ids_flattened.append(torch.cat((cur_sample_ids, pad_sample_ids), dim=0))

        final_full_sample_ids = torch.stack(full_sample_ids_flattened, dim=0).view(batch_size * FLAGS.rl_sample_size, -1)
        sample_scores = self.reward_calculator.compute_per_step_rewards(
            gold_answers, partial_outputs=samples, terminal_reward_only=True
        )
        sample_scores_tensor = torch.tensor(sample_scores, dtype=torch.float64, device=self.policy_lm.device)
        sample_scores_tensor = sample_scores_tensor.view(batch_size, FLAGS.rl_sample_size)

        # Making sure to be between zero and one.
        normalized_scores = normalize_signals(sample_scores_tensor, normalization_type="linear")
        max_values, max_indices = torch.max(normalized_scores, dim=1, keepdim=True)
        min_values, min_indices = torch.min(normalized_scores, dim=1, keepdim=True)

        # Accept or reject the samples.
        reward_chosen_log_probs = torch.gather(sequence_log_probs, dim=1, index=max_indices)
        reward_rejected_log_probs = torch.gather(sequence_log_probs, dim=1, index=min_indices)

        reference_free = True
        if FLAGS.preference_optimization_with_reference:
            ref_sequence_log_ps, _ = self.find_reference_scores(batch, final_full_sample_ids)
            reward_chosen_ref_log_probs = torch.gather(ref_sequence_log_ps, dim=1, index=max_indices)
            reward_rejected_ref_log_probs = torch.gather(ref_sequence_log_ps, dim=1, index=min_indices)
            reference_free = False

        preference_losses = preference_loss(
            policy_chosen_logps=reward_chosen_log_probs,
            policy_rejected_logps=reward_rejected_log_probs,
            beta=FLAGS.preference_beta,
            label_smoothing=FLAGS.preference_label_smoothing,
            ipo=FLAGS.preference_ipo,
            reference_free=reference_free,
            reference_chosen_logps=reward_chosen_ref_log_probs if not reference_free else None,
            reference_rejected_logps=reward_rejected_ref_log_probs if not reference_free else None,
        )
        return torch.mean(preference_losses)

    def maximum_marginal_likelihood_loss(
        self,
        batch: torch.utils.data.Dataset,
        iterative_finetuning: bool = False,
        reinforce_terminal_reward: bool = False,
        mixed: bool = False,
    ) -> torch.Tensor:
        """Use maximum marginal likelihood training to compute the loss."""
        assert FLAGS.rl_sample_size > 1
        num_iterative_calls = FLAGS.rl_sample_size // FLAGS.iterative_chunk_size
        generations = []
        final_log_ps = []
        for call_idx in range(num_iterative_calls):
            llm_generation_outputs: List[LLMGenerationOutput] = self.policy_lm.generation_pass(
                batch,
                top_p=FLAGS.train_top_p,
                temperature=FLAGS.train_temperature,
                num_return_sequences=FLAGS.iterative_chunk_size,
                to_train=True,
                use_cache=True,
                per_step_scores=False,
                iterative_rl_sampling=False,
            )
            generations_per_call = llm_generation_outputs[0].predictions_str
            final_log_ps_per_call = llm_generation_outputs[0].final_log_ps
            batch_generations_per_call = []
            batch_size = len(generations_per_call) // FLAGS.iterative_chunk_size
            for b_idx in range(batch_size):
                batch_generations_per_call.append(
                    [generations_per_call[b_idx * FLAGS.iterative_chunk_size : (b_idx + 1) * FLAGS.iterative_chunk_size]]
                )
            final_log_ps.append(final_log_ps_per_call.view(batch_size, FLAGS.iterative_chunk_size))
            generations.append(batch_generations_per_call)

        sequence_log_probs = torch.cat(final_log_ps, dim=1)
        batch_size = sequence_log_probs.size()[0]
        # Compute the rewards.
        gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
        samples = []
        for b_idx in range(batch_size):
            sample_arr = []
            for call_idx in range(num_iterative_calls):
                samples_strings = generations[call_idx][b_idx][0]
                sample_arr.extend(samples_strings)

            # Add extra third dimension.
            sample_arr_extended = [[each] for each in sample_arr]
            samples.append(sample_arr_extended)

        sample_scores = self.reward_calculator.compute_per_step_rewards(
            gold_answers, partial_outputs=samples, terminal_reward_only=True
        )
        sample_scores_tensor = torch.tensor(sample_scores, dtype=torch.float64, device=self.policy_lm.device)
        sample_scores_tensor = sample_scores_tensor.view(batch_size, FLAGS.rl_sample_size)

        # Making sure to be between zero and one.
        normalized_scores = normalize_signals(sample_scores_tensor, normalization_type="linear")
        mml_loss = 0.0
        iterative_loss = 0.0
        reinforce_loss = 0.0
        if ((not iterative_finetuning) and (not reinforce_terminal_reward)) or mixed:
            # These are full sequence returns.
            # This is the MML objective.
            if FLAGS.mml_version == "version_1":
                log_of_scores = torch.log(normalized_scores + 1e-12)
                mml_loss = -torch.mean(torch.logsumexp(sequence_log_probs + log_of_scores, dim=1), dim=0)
            elif FLAGS.mml_version == "version_2":
                mml_loss = -torch.mean(torch.logsumexp(sequence_log_probs + normalized_scores, dim=1), dim=0)

        elif (iterative_finetuning and (not reinforce_terminal_reward)) or mixed:
            # This is iterative fine-tuning.
            # Find the sample with the highest return.
            # These are full sequence returns.
            max_values, max_indices = torch.max(normalized_scores, dim=1, keepdim=True)

            # Use if it is a good sample.
            return_masks = (max_values > 0.5).float()
            selected_log_probs = torch.gather(sequence_log_probs, dim=1, index=max_indices) * return_masks
            iterative_loss = -torch.mean(
                selected_log_probs.view(
                    batch_size,
                ),
                dim=0,
            )

        elif ((not iterative_finetuning) and reinforce_terminal_reward) or mixed:
            # reinforce with terminal reward without per-step rewards.
            reinforce_loss = -torch.mean(sequence_log_probs * normalized_scores)

        return reinforce_loss + mml_loss + iterative_loss

    def hard_em_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Use maximum marginal likelihood training to compute the loss."""
        llm_generation_outputs: List[LLMGenerationOutput] = self.policy_lm.generation_pass(
            batch,
            top_p=FLAGS.test_top_p,
            temperature=FLAGS.test_temperature,
            num_return_sequences=1,
            to_train=True,
            use_cache=True,
            per_step_scores=False,
            iterative_rl_sampling=False,
        )
        generations = llm_generation_outputs[0].predictions_str
        final_log_ps = llm_generation_outputs[0].final_log_ps
        batch_size = len(generations)
        sequence_log_probs = final_log_ps.view(
            batch_size,
        )
        loss = -torch.mean(sequence_log_probs, dim=0)
        return loss

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Based on the train objective type, call the right loss function."""

        if self.objective_type == "teacher_forcing":
            return self.teacher_forcing_loss(batch)

        elif self.objective_type == "mml":
            return self.maximum_marginal_likelihood_loss(batch)

        elif self.objective_type == "hard_em":
            return self.hard_em_loss(batch)

        elif self.objective_type == "iterative_finetuning":
            return self.maximum_marginal_likelihood_loss(batch, iterative_finetuning=True)

        elif self.objective_type == "reinforce_terminal_reward_version_1":
            return self.maximum_marginal_likelihood_loss(batch, reinforce_terminal_reward=True)

        elif self.objective_type == "reinforce_terminal_reward_version_2":
            return self.reinforce_loss(batch, terminal_reward_only=True)

        elif self.objective_type == "mml_iterative_reinforce":
            return self.maximum_marginal_likelihood_loss(batch, mixed=True)

        elif self.objective_type == "reinforce":
            return self.reinforce_loss(batch)

        elif self.objective_type == "preference_optimization":
            return self.preference_optimization_loss(batch)

        # elif self.objective_type == "teacher_forcing_reinforce":
        #     return self.reinforce_loss(batch) + self.teacher_forcing_loss(batch)

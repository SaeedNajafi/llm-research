"""The main module for different objectives to train the policy (llm)."""

from itertools import chain
from typing import Any, List, Optional

import torch
from absl import flags, logging
from torch.utils.data import DataLoader

from src.llm import LLM, LLMGenerationOutput
from src.metrics import RewardCalculator
from src.utils.general_utils import DictDataset
from src.utils.rl_utils import compute_entropy_loss, form_returns

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "rl_sample_size", 8, "The number of samples to generate from the policy used for both on/off-policy learnings."
)
flags.DEFINE_integer(
    "iterative_chunk_size", 2, "The number of return sequences to generate from LLM at the same time in parallel."
)
flags.DEFINE_boolean("with_baseline", False, "Whether to use the baseline reward in RL objective.")
flags.DEFINE_string("reward_normalization_type", "zscore", "zscore | mml_normalize | normalize | rloo_normalize | no_normalize")
flags.DEFINE_float("baseline_momentum", 0.9, "momentum used to compute the average reward in the RL baseline.")
flags.DEFINE_boolean("compute_per_step_entropy", False, "Whether to add per-step entropy to the loss in RL training.")
flags.DEFINE_float("entropy_coef", 0.1, "Coefficient used to mix per-step entropy loss in RL training.")
flags.DEFINE_string("objective_type", "reinforce", "Different objectives to get the loss for training the llm.")
flags.DEFINE_boolean(
    "include_policy_ref_kl", False, "Whether to apply the KL divergence between the policy and the reference policy."
)
flags.DEFINE_float(
    "policy_ref_kl_coef", 0.1, "Coefficient to apply the KL divergence between the policy and the reference policy."
)


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
            # For simple, average return baseline.
            self.baseline_reward = 0.0
        self.objective_type = objective_type
        self.reward_calculator = RewardCalculator(reward_name, weights_base_folder)

    def teacher_forcing_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        return self.policy_lm.train(batch)

    def sample_and_generate_details(self, batch: torch.utils.data.Dataset) -> Any:
        """Compute per-step information while sampling."""
        assert FLAGS.rl_sample_size > 1
        num_iterative_calls = FLAGS.rl_sample_size // FLAGS.iterative_chunk_size
        generations = []
        partial_generations = []
        final_log_ps = []
        actual_lens = []
        labels_to_consider = []
        token_final_log_ps = []
        logits = []
        for call_idx in range(num_iterative_calls):
            llm_generation_outputs: List[LLMGenerationOutput] = self.policy_lm.generation_pass(
                batch,
                top_p=FLAGS.train_top_p,
                temperature=FLAGS.train_temperature,
                num_return_sequences=FLAGS.iterative_chunk_size,
                to_train=True,
                use_cache=True,
                per_step_scores=True,
                iterative_rl_sampling=False,
                generate_partial_sequences=True,
            )
            generations_per_call = llm_generation_outputs[0].predictions_str
            final_log_ps_per_call = llm_generation_outputs[0].final_log_ps
            actual_lens_per_call = llm_generation_outputs[0].actual_lens
            labels_to_consider_per_call = llm_generation_outputs[0].labels_to_consider
            token_final_log_ps_per_call = llm_generation_outputs[0].token_final_log_ps
            partial_generations_per_call = llm_generation_outputs[0].partially_generated_sequences
            logits_per_call = llm_generation_outputs[0].logits
            _, seq_len, vocab_size = logits_per_call.size()
            batch_generations_per_call = []
            batch_partial_generations_per_call = []
            batch_size = len(generations_per_call) // FLAGS.iterative_chunk_size
            for b_idx in range(batch_size):
                batch_generations_per_call.append(
                    [generations_per_call[b_idx * FLAGS.iterative_chunk_size : (b_idx + 1) * FLAGS.iterative_chunk_size]]
                )
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
            logits.append(logits_per_call.view(batch_size, FLAGS.iterative_chunk_size, seq_len, vocab_size))
            generations.append(batch_generations_per_call)
            partial_generations.append(batch_partial_generations_per_call)
        sequence_log_probs = torch.cat(final_log_ps, dim=1)
        actual_lens = torch.cat(actual_lens, dim=1)
        batch_size = sequence_log_probs.size()[0]
        samples = []
        final_labels_to_consider = []
        token_log_ps = []
        final_logits = []
        partial_samples = []
        for b_idx in range(batch_size):
            sample_arr = []
            labels_to_consider_arr = []
            token_log_ps_arr = []
            logits_arr = []
            partial_samples_arr = []
            for call_idx in range(num_iterative_calls):
                sample_arr.extend(generations[call_idx][b_idx][0])
                for chunk_idx in range(FLAGS.iterative_chunk_size):
                    labels_to_consider_arr.append(labels_to_consider[call_idx][b_idx, chunk_idx, :])
                    token_log_ps_arr.append(token_final_log_ps[call_idx][b_idx, chunk_idx, :])
                    logits_arr.append(logits[call_idx][b_idx, chunk_idx, :, :])
                    partial_samples_arr.append(partial_generations[call_idx][b_idx][0][chunk_idx])

            final_labels_to_consider.append(labels_to_consider_arr)
            token_log_ps.append(token_log_ps_arr)
            final_logits.append(logits_arr)
            samples.append(sample_arr)
            partial_samples.append(partial_samples_arr)

        return_data = {
            "samples": samples,
            "sequence_log_probs": sequence_log_probs,
            "actual_lens": actual_lens,
            "partial_samples": partial_samples,
            "labels_to_consider": final_labels_to_consider,
            "token_log_ps": token_log_ps,
            "logits": final_logits,
        }

        return return_data

    def reinforce_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Use reinforce with per-step rewards to compute the loss."""
        sample_data = self.sample_and_generate_details(batch)
        batch_size = len(batch["gold_answers"])
        # These are full sequence returns.
        # Compute the rewards.
        gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
        per_step_rewards = self.reward_calculator.compute_per_step_rewards(
            gold_answers,
            sample_data["partial_samples"],
            output_template=self.policy_lm.output_template,
            terminal_reward_only=True,
        )
        normalized_per_step_rewards = self.normalize_signals(signals=per_step_rewards, terminal_reward_only=True)
        if FLAGS.include_policy_ref_kl:
            # Compute log likelihood of the reference model for the samples.
            cleaned_samples = []
            for per_example_samples in sample_data["partial_samples"]:
                per_example_cleaned_samples = []
                for sample_sequence in per_example_samples:
                    # Last complete one!
                    sample_output = sample_sequence[-1]
                    per_example_cleaned_samples.append(sample_output)
                cleaned_samples.append(per_example_cleaned_samples)

            num_samples_per_example = FLAGS.rl_sample_size
            input_texts = batch["texts"]
            row_ids = batch["row_ids"]
            expanded_input_texts = list(chain.from_iterable([[text] * num_samples_per_example for text in input_texts]))
            expanded_row_ids = list(chain.from_iterable([[row_id] * num_samples_per_example for row_id in row_ids]))
            flattened_sample_outputs = list(chain.from_iterable(cleaned_samples))
            data = self.ref_policy_lm.prepare_text_for_train(
                texts=expanded_input_texts, output_texts=flattened_sample_outputs, row_ids=expanded_row_ids
            )
            dataset = DictDataset(data)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=len(dataset),
                num_workers=1,
            )
            for ref_batch in dataloader:
                _, ref_token_log_probs, _, _ = self.ref_policy_lm.train(ref_batch, per_step_scores=True, to_train=False)

            # the sequence length in the token_log_probs has been shifted to the right by 1 position.
            ref_token_log_probs = ref_token_log_probs.view(batch_size, FLAGS.rl_sample_size, -1)
            ref_token_log_probs_arr = []
            for b_idx in range(batch_size):
                ref_token_log_probs_arr_per_example = []
                for s_idx in range(FLAGS.rl_sample_size):
                    ref_token_log_probs_arr_per_example_per_sample = []
                    ref_token_log_prob_sequence = ref_token_log_probs[b_idx, s_idx, :]
                    for ref_token_log_prob in ref_token_log_prob_sequence:
                        if ref_token_log_prob.item() != 0.0:
                            ref_token_log_probs_arr_per_example_per_sample.append(ref_token_log_prob.item())
                    ref_token_log_probs_arr_per_example.append(ref_token_log_probs_arr_per_example_per_sample)
                ref_token_log_probs_arr.append(ref_token_log_probs_arr_per_example)

            # normalize the ref log probs.
            normalized_ref_token_log_probs_arr = self.normalize_signals(signals=ref_token_log_probs_arr)
            # add ref_log_prob as a new reward.
            for b_idx in range(batch_size):
                for s_idx in range(FLAGS.rl_sample_size):
                    sequence_rewards = normalized_per_step_rewards[b_idx][s_idx]
                    ref_kl_sequence_rewards = normalized_ref_token_log_probs_arr[b_idx][s_idx]
                    try:
                        for seq_idx in range(len(sequence_rewards)):
                            normalized_per_step_rewards[b_idx][s_idx][seq_idx] += (
                                FLAGS.policy_ref_kl_coef * ref_kl_sequence_rewards[seq_idx]
                            )
                    except Exception:
                        print("saeed")
                        print(len(sequence_rewards))
                        print(len(ref_kl_sequence_rewards))
                        print(len(ref_token_log_probs_arr[b_idx][s_idx]))
                        print(ref_token_log_probs.size())
                        print(sample_data["actual_lens"])
                        exit()

        returns = form_returns(normalized_per_step_rewards)
        normalized_returns = self.normalize_signals(returns)
        objective = 0.0
        if FLAGS.with_baseline:
            current_batch_sample_average_rewards = 0.0
        for b_idx in range(batch_size):
            for sample_idx in range(FLAGS.rl_sample_size):
                sequence_token_log_ps = sample_data["token_log_ps"][b_idx][sample_idx]
                sequence_returns = normalized_returns[b_idx][sample_idx]
                if FLAGS.with_baseline:
                    current_batch_sample_average_rewards += torch.mean(sequence_returns, dim=0)
                    sequence_returns = sequence_returns - self.baseline_reward
                objective += torch.sum(sequence_token_log_ps * sequence_returns)

        if FLAGS.with_baseline:
            new_baseline_reward = (current_batch_sample_average_rewards / FLAGS.rl_sample_size) / batch_size
            self.baseline_reward = (
                FLAGS.baseline_momentum * self.baseline_reward + (1.0 - FLAGS.baseline_momentum) * new_baseline_reward
            )
            msg = f"\nbaseline reward used: {self.baseline_reward}"
            logging.info(msg)

        loss = -(objective / FLAGS.rl_sample_size) / batch_size

        # Compute the per-step entropy if requested.
        if FLAGS.compute_per_step_entropy:
            entropy_loss = compute_entropy_loss(
                sample_data["labels_to_consider"],
                sample_data["actual_lens"],
                sample_data["token_log_ps"],
                sample_data["logits"],
            )
            msg = f"\nentropy loss computed: {entropy_loss}"
            logging.info(msg)
            coefficient = FLAGS.entropy_coef
            if FLAGS.include_policy_ref_kl:
                # necessary to train with kl penalty between ref and policy.
                coefficient += FLAGS.policy_ref_kl_coef
            loss += coefficient * entropy_loss

        return loss

    def maximum_marginal_likelihood_loss(
        self,
        batch: torch.utils.data.Dataset,
        iterative_finetuning: bool = False,
        reinforce_terminal_reward_only: bool = False,
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
                sample_arr.extend(generations[call_idx][b_idx][0])
            samples.append(sample_arr)

        sample_returns = self.reward_calculator.compute_rewards(gold_answers, samples)
        if (not iterative_finetuning) and (not reinforce_terminal_reward_only):
            # These are full sequence returns.
            returns = torch.tensor(sample_returns, dtype=torch.float64, device=self.policy_lm.device)
            # This is the MML objective.
            log_of_returns = torch.log(returns + 1e-12)
            loss = -torch.mean(torch.logsumexp(sequence_log_probs + log_of_returns, dim=1), dim=0)
            return loss
        elif reinforce_terminal_reward_only:
            normalized_returns = self.normalize_signals(signals=[sample_returns])
            normalized_returns = normalized_returns[0]  # remove extra []
            objective = 0.0
            for b_idx in range(batch_size):
                example_sequence_log_ps = sequence_log_probs[b_idx, :]
                example_returns = normalized_returns[b_idx]
                objective += torch.mean(example_sequence_log_ps * example_returns)
            loss = -objective / batch_size
            return loss
        else:
            # This is iterative fine-tuning.
            # Find the sample with the highest return.
            # These are full sequence returns.
            returns = torch.tensor(sample_returns, dtype=torch.float64, device=self.policy_lm.device)
            max_values, max_indices = torch.max(returns, dim=1, keepdim=True)
            return_masks = (max_values > 0.5).float()
            selected_log_probs = torch.gather(sequence_log_probs, dim=1, index=max_indices) * return_masks
            loss = -torch.mean(
                selected_log_probs.view(
                    batch_size,
                ),
                dim=0,
            )
            return loss

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

        elif self.objective_type == "reinforce_terminal_reward":
            return self.maximum_marginal_likelihood_loss(batch, reinforce_terminal_reward_only=True)

        elif self.objective_type == "reinforce":
            return self.reinforce_loss(batch)

        elif self.objective_type == "teacher_forcing_reinforce":
            return self.reinforce_loss(batch) + self.teacher_forcing_loss(batch)

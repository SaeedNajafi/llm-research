"""The main module for different objectives to train the policy (llm)."""

from typing import List, Optional

import torch
from absl import flags

from src.llm import LLM, LLMGenerationOutput
from src.metrics import RewardCalculator

# from torch.distributions import Categorical


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


class LossCalculator:

    def __init__(
        self,
        policy_lm: LLM,
        objective_type: str,
        reward_name: str,
        value_lm: Optional[LLM] = None,
        ref_policy_lm: Optional[LLM] = None,
    ):
        super().__init__()
        self.policy_lm = policy_lm
        self.value_lm = value_lm
        self.ref_policy_lm = ref_policy_lm
        if FLAGS.with_baseline:
            self.baseline_reward = 0.0
        self.objective_type = objective_type
        self.reward_calculator = RewardCalculator(reward_name=reward_name)

    def normalize_signals(self, signals: List[List[List[float]]], flat_signals: List[float]) -> List[List[torch.Tensor]]:
        """Zscore or normalize between [-1, 1]."""
        flat_signals_t = torch.tensor(flat_signals, dtype=torch.float64, device=self.policy_lm.device)
        mean_s = flat_signals_t.mean()
        std_s = flat_signals_t.std()
        max_s = flat_signals_t.max()
        min_s = flat_signals_t.min()
        batch_size = len(signals)
        normalized_signals_arr = []
        for b_idx in range(batch_size):
            sample_size = len(signals[b_idx])
            sample_normalized_signals = []
            for sample_idx in range(sample_size):
                sample_signals = signals[b_idx][sample_idx]
                sample_signals = torch.tensor(sample_signals, dtype=torch.float64, device=self.policy_lm.device)
                if FLAGS.reward_normalization_type == "zscore":
                    normalized_signals = (sample_signals - mean_s) / (std_s + 1e-12)
                elif FLAGS.reward_normalization_type == "normalize":
                    normalized_signals = 2 * (sample_signals - min_s) / (max_s - min_s + 1e-12) - 1.0
                elif FLAGS.reward_normalization_type == "no_normalize":
                    normalized_signals = sample_signals
                sample_normalized_signals.append(normalized_signals)
            normalized_signals_arr.append(sample_normalized_signals)

        return normalized_signals_arr

    def teacher_forcing_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        return self.policy_lm.train(batch)

    def reinforce_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Use reinforce with per-step rewards to compute the loss."""
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

        # These are full sequence returns.
        # Compute the rewards.
        gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
        per_step_returns, flat_returns = self.reward_calculator.compute_returns(
            gold_answers, partial_samples, output_template=self.policy_lm.output_template
        )
        normalized_returns = self.normalize_signals(signals=per_step_returns, flat_signals=flat_returns)
        objective = 0.0
        for b_idx in range(batch_size):
            for sample_idx in range(FLAGS.rl_sample_size):
                sequence_token_log_ps = token_log_ps[b_idx][sample_idx]
                sequence_returns = normalized_returns[b_idx][sample_idx]
                objective += torch.sum(sequence_token_log_ps * sequence_returns)
        loss = -(objective / FLAGS.rl_sample_size) / batch_size

        #     # Subtract the baseline value of the rewards.
        #     if FLAGS.with_baseline:
        #         # mean pulling over the best rewards per example.
        #         max_normalized_rewards, _ = torch.max(normalized_rewards, dim=1, keepdim=True)
        #         normalized_rewards -= self.baseline_reward

        #         new_baseline_reward = torch.mean(torch.mean(max_normalized_rewards, dim=1), dim=0)
        #         self.baseline_reward = (
        #             FLAGS.baseline_momentum * self.baseline_reward + (1.0 - FLAGS.baseline_momentum) * new_baseline_reward
        #         )

        #     # Compute the per-step entropy if requested.
        #     if FLAGS.compute_per_step_entropy:
        #         entropy_masks = torch.where(labels_to_consider == -100, 0, 1)
        #         distribution = Categorical(logits=logits)
        #         sequence_entropy = torch.sum(distribution.entropy() * entropy_masks, dim=1) / actual_lens
        #         sequence_entropy = sequence_entropy.view(batch_size, FLAGS.rl_sample_size)

        #     if FLAGS.compute_per_step_entropy:
        #         entropy_loss_part_one = -torch.mean(torch.mean(sequence_log_probs * sequence_entropy.detach(), dim=1), dim=0)
        #         entropy_loss_part_two = -torch.mean(torch.mean(sequence_entropy, dim=1), dim=0)
        #         entropy_loss = entropy_loss_part_one + entropy_loss_part_two
        #         loss += FLAGS.entropy_coef * entropy_loss

        #     return loss
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
            # This is reinforce style training.
            flatten_returns = []
            for batch_returns in sample_returns:
                flatten_returns.extend(batch_returns)
            normalized_returns = self.normalize_signals(signals=[sample_returns], flat_signals=flatten_returns)
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

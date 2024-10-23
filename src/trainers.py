"""The main module for different objectives to train the policy (llm)."""

from typing import List, Optional

import torch
from absl import flags
from torch.distributions import Categorical

from src.llm import LLM
from src.metrics import RewardCalculator
from src.utils.rl_utils import normalize, rloo_normalize, z_scoring

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

    def normalize_rewards(self, sample_output_rewards: List[List[float]]) -> torch.Tensor:
        """Zscore or normalize between [-1, 1] or MML style normalization."""
        rewards = torch.tensor(sample_output_rewards, device=self.policy_lm.device)
        if FLAGS.reward_normalization_type == "zscore":
            return z_scoring(rewards)

        elif FLAGS.reward_normalization_type == "normalize":
            return normalize(rewards)

        elif FLAGS.reward_normalization_type == "rloo_normalize":
            return rloo_normalize(rewards)

        elif FLAGS.reward_normalization_type == "no_normalize":
            return rewards

    def teacher_forcing_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        return self.policy_lm.train(batch)

    def reinforce_loss(
        self,
        batch: torch.utils.data.Dataset,
        sample_outputs: List[List[str]],
        sample_output_rewards: List[List[float]],
        compute_per_step_entropy: bool = False,
    ) -> torch.Tensor:
        """We have to feed the input along with new sampled outputs to train
        the policy."""

        # Check what is the error at the end of the training and evaluation.

        # Check why does it require a very large memory.

        # Implement how you can provide KL penalty with respect to the reference policy.

        # Test the implementations in a task.

        # TODO: 1 - Implement having a reference model on another GPU for inference Only.

        # 3 - Implement the approximate KL used in PPO's original paper.

        # 5 - Implement the Genealized Advantage Function and Consider the Value Network with a loss to train it.

        # Provide Implementation for PPO and A2C.

        # Switch to NarrativeQA dataset.

        # Exactly Implement this: https://github.com/allenai/RL4LMs/blob/main/rl4lms/algorithms/ppo/ppo.py

        # Exactly Implement this: https://github.com/allenai/RL4LMs/blob/main/rl4lms/algorithms/a2c/a2c.py

        # TODO: Research Questions:
        # Should we consider the KL div similar to RLHF?
        # Should we switch to MiniLLM idea to approximate the reverse KL?
        # Should we implement DPO?
        # Should we implement the soft q-learning?
        # Should we implement the soft actor-critic?
        pass


    # def on_policy_rl_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
    #     """This is the function to sample from the same policy and train it
    #     with RL loss."""
    #     generations, final_log_ps, token_final_log_ps, actual_lens, logits, labels_to_consider = self.policy_lm.generation_pass(
    #         batch,
    #         top_p=FLAGS.train_top_p,
    #         temperature=FLAGS.train_temperature,
    #         num_return_sequences=FLAGS.rl_sample_size,
    #         to_train=True,
    #         use_cache=True,
    #     )
    #     print(generations)
    #     print(batch["gold_answers"])
    #     print("\n\n##")
    #     cleaned_samples = [text.removeprefix("assistant\n\n").removeprefix("Final Answer: ") for text in generations]
    #     batch_size = len(cleaned_samples) // FLAGS.rl_sample_size
    #     sequence_log_probs = final_log_ps.view(batch_size, FLAGS.rl_sample_size)

    #     # Compute the rewards.
    #     gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
    #     samples = [
    #         cleaned_samples[b_idx * FLAGS.rl_sample_size : (b_idx + 1) * FLAGS.rl_sample_size] for b_idx in range(batch_size)
    #     ]
    #     sample_rewards = self.reward_calculator.compute_rewards(gold_answers, samples)
    #     rewards = torch.tensor(sample_rewards, dtype=torch.float64, device=self.policy_lm.device)

    #     # Normalize the rewards.
    #     normalized_rewards = self.normalize_rewards(rewards)

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

    #     # Compute the losses.
    #     if FLAGS.reward_normalization_type == "mml_normalize":
    #         loss = -torch.mean(torch.sum(sequence_log_probs * normalized_rewards, dim=1), dim=0)
    #     else:
    #         loss = -torch.mean(torch.mean(sequence_log_probs * normalized_rewards, dim=1), dim=0)

    #     if FLAGS.compute_per_step_entropy:
    #         entropy_loss_part_one = -torch.mean(torch.mean(sequence_log_probs * sequence_entropy.detach(), dim=1), dim=0)
    #         entropy_loss_part_two = -torch.mean(torch.mean(sequence_entropy, dim=1), dim=0)
    #         entropy_loss = entropy_loss_part_one + entropy_loss_part_two
    #         loss += FLAGS.entropy_coef * entropy_loss

    #     return loss

    def maximum_marginal_likelihood_loss(
        self, batch: torch.utils.data.Dataset, iterative_finetuning: bool = False
    ) -> torch.Tensor:
        """Use maximum marginal likelihood training to compute the loss."""
        assert FLAGS.rl_sample_size > 1
        num_iterative_calls = FLAGS.rl_sample_size // FLAGS.iterative_chunk_size
        generations = []
        final_log_ps = []
        for call_idx in range(num_iterative_calls):
            llm_generation_outputs = self.policy_lm.generation_pass(
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

        # These are full sequence returns.
        sample_returns = self.reward_calculator.compute_rewards(gold_answers, samples)
        returns = torch.tensor(sample_returns, dtype=torch.float64, device=self.policy_lm.device)
        if not iterative_finetuning:
            # This is the MML objective.
            log_of_returns = torch.log(returns + 1e-12)
            loss = -torch.mean(torch.logsumexp(sequence_log_probs + log_of_returns, dim=1), dim=0)
            return loss
        else:
            # This is iterative fine-tuning.
            # Find the sample with the highest return.
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
        llm_generation_outputs = self.policy_lm.generation_pass(
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

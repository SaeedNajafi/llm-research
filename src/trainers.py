"""The main module for different objectives to train the policy (llm)."""

from itertools import chain
from typing import Any, List, Optional

import torch
from absl import flags
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from src.llm import LLM
from src.metrics import RewardCalculator
from src.utils.general_utils import DictDataset
from src.utils.rl_utils import mml_normalize, normalize, rloo_normalize, z_scoring

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "rl_sample_size", 4, "The number of samples to generate from the policy used for both on/off-policy learnings."
)
flags.DEFINE_boolean("iterative_computation", False, "Whether to compute the loss per example; useful for avoiding OOM.")
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

    def compute_policy_log_probs(
        self,
        input_texts: List[str],
        row_ids: List[str],
        sample_outputs: List[List[str]],
        compute_per_step_entropy: bool = False,
    ) -> Any:
        """Feed the input along with the sampled output to compute the log
        probability of the policy for these actions."""
        batch_size = len(input_texts)
        if FLAGS.iterative_computation:
            # Useful when we cannot fit all samples for mini-batch at the same time in GPU.
            sequence_log_probs_arr = []
            token_log_probs_arr = []
            logits_arr = []
            masked_labels_arr = []
            for batch_idx, input_text in enumerate(input_texts):
                row_id = row_ids[batch_idx]
                example_samples = sample_outputs[batch_idx]
                data = self.policy_lm.prepare_text_for_train(
                    texts=[input_text] * len(example_samples),
                    output_texts=example_samples,
                    row_ids=[row_id] * len(example_samples),
                )
                dataset = DictDataset(data)
                dataloader = DataLoader(
                    dataset,
                    shuffle=False,
                    batch_size=len(dataset),
                    num_workers=1,
                )
                for batch in dataloader:
                    sequence_log_probs, token_log_probs, logits, masked_labels = self.policy_lm.train(
                        batch, per_step_scores=True
                    )

                sequence_log_probs_arr.append(sequence_log_probs)
                token_log_probs_arr.append(token_log_probs)
                logits_arr.append(logits)
                masked_labels_arr.append(masked_labels)

            return sequence_log_probs_arr, token_log_probs_arr, logits_arr, masked_labels_arr

        else:
            expanded_input_texts = list(chain.from_iterable([[text] * FLAGS.rl_sample_size for text in input_texts]))
            expanded_row_ids = list(chain.from_iterable([[row_id] * FLAGS.rl_sample_size for row_id in row_ids]))
            flattened_sample_outputs = list(chain.from_iterable(sample_outputs))
            data = self.policy_lm.prepare_text_for_train(
                texts=expanded_input_texts, output_texts=flattened_sample_outputs, row_ids=expanded_row_ids
            )
            dataset = DictDataset(data)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=1,
            )
            sequence_log_probs_arr = []
            sequence_entropy_arr = []
            if compute_per_step_entropy:
                for batch in dataloader:
                    sequence_log_probs_sample, _, logits_sample, masked_labels_sample = self.policy_lm.train(
                        batch, per_step_scores=True
                    )
                    sequence_log_probs_arr.append(sequence_log_probs_sample)

                    # Compute per-step entropy.
                    entropy_masks = torch.where(masked_labels_sample == -100, 0, 1)
                    actual_lens = torch.sum(entropy_masks, dim=1)
                    distribution = Categorical(logits=logits_sample)
                    sequence_entropy_sample = torch.sum(distribution.entropy() * entropy_masks, dim=1) / actual_lens
                    sequence_entropy_arr.append(sequence_entropy_sample)

                sequence_log_probs = torch.cat(sequence_log_probs_arr, dim=0)
                sequence_entropy = torch.cat(sequence_entropy_arr, dim=0)

                sequence_log_probs = sequence_log_probs.view(batch_size, FLAGS.rl_sample_size)
                sequence_entropy = sequence_entropy.view(batch_size, FLAGS.rl_sample_size)
                return sequence_log_probs, sequence_entropy
            else:
                for batch in dataloader:
                    sequence_log_probs_sample = self.policy_lm.train(batch, per_step_scores=False)
                    sequence_log_probs_arr.append(sequence_log_probs_sample)

                sequence_log_probs = torch.cat(sequence_log_probs_arr, dim=0)
                sequence_log_probs = sequence_log_probs.view(batch_size, FLAGS.rl_sample_size)
                return sequence_log_probs, None

    def normalize_rewards(self, sample_output_rewards: List[List[float]]) -> torch.Tensor:
        """Zscore or normalize between [-1, 1] or MML style normalization."""
        rewards = torch.tensor(sample_output_rewards, device=self.policy_lm.device)
        if FLAGS.reward_normalization_type == "zscore":
            return z_scoring(rewards)

        elif FLAGS.reward_normalization_type == "normalize":
            return normalize(rewards)

        elif FLAGS.reward_normalization_type == "mml_normalize":
            return mml_normalize(rewards)

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
        original_input_texts = batch["texts"]
        original_row_ids = batch["row_ids"]

        sequence_log_probs, sequence_entropy = self.compute_policy_log_probs(
            input_texts=original_input_texts,
            row_ids=original_row_ids,
            sample_outputs=sample_outputs,
            compute_per_step_entropy=compute_per_step_entropy,
        )
        sample_output_rewards = torch.tensor(sample_output_rewards, dtype=torch.float64, device=self.policy_lm.device)
        normalized_rewards = self.normalize_rewards(sample_output_rewards)
        if FLAGS.with_baseline:
            # mean pulling over the best rewards per example.
            max_normalized_rewards, _ = torch.max(normalized_rewards, dim=1, keepdim=True)
            normalized_rewards -= self.baseline_reward

            new_baseline_reward = torch.mean(torch.mean(max_normalized_rewards, dim=1), dim=0)
            self.baseline_reward = (
                FLAGS.baseline_momentum * self.baseline_reward + (1.0 - FLAGS.baseline_momentum) * new_baseline_reward
            )

        if FLAGS.iterative_computation:
            # with iterative computation, we are dealing with a list of tensors.
            # sequence length might be different between examples.
            sequence_log_probs = torch.cat(sequence_log_probs, dim=0)
            batch_size = len(original_input_texts)
            # second dimension is the number of samples per example.
            sequence_log_probs = sequence_log_probs.view(batch_size, -1)

        if FLAGS.reward_normalization_type == "mml_normalize":
            loss = -torch.mean(torch.sum(sequence_log_probs * normalized_rewards, dim=1), dim=0)
        else:
            loss = -torch.mean(torch.mean(sequence_log_probs * normalized_rewards, dim=1), dim=0)

        if compute_per_step_entropy:
            entropy_loss_part_one = -torch.mean(torch.mean(sequence_log_probs * sequence_entropy.detach(), dim=1), dim=0)
            entropy_loss_part_two = -torch.mean(torch.mean(sequence_entropy, dim=1), dim=0)
            entropy_loss = entropy_loss_part_one + entropy_loss_part_two
            loss += FLAGS.entropy_coef * entropy_loss

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
        return loss

    def on_policy_rl_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """This is the function to sample from the same policy and train it
        with RL loss."""
        generations, _ = self.policy_lm.generation_pass(
            batch, top_p=FLAGS.train_top_p, temperature=FLAGS.train_temperature, num_return_sequences=FLAGS.rl_sample_size
        )
        cleaned_samples = [text.removeprefix("assistant\n\n").removeprefix("Final Answer: ") for text in generations]
        print(cleaned_samples)
        templated_samples = [
            self.policy_lm.output_template.format(output=f"Final Answer: {sample}") for sample in cleaned_samples
        ]
        batch_size = len(templated_samples) // FLAGS.rl_sample_size
        sample_gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
        sample_outputs = [
            templated_samples[b_idx * FLAGS.rl_sample_size : (b_idx + 1) * FLAGS.rl_sample_size] for b_idx in range(batch_size)
        ]
        sample_clean_outputs = [
            cleaned_samples[b_idx * FLAGS.rl_sample_size : (b_idx + 1) * FLAGS.rl_sample_size] for b_idx in range(batch_size)
        ]
        sample_rewards = self.reward_calculator.compute_rewards(sample_gold_answers, sample_clean_outputs)
        loss = self.reinforce_loss(
            batch, sample_outputs, sample_rewards, compute_per_step_entropy=FLAGS.compute_per_step_entropy
        )
        return loss

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Based on the train objective type, call the right loss function."""

        if self.objective_type == "teacher_forcing":
            return self.teacher_forcing_loss(batch)

        elif self.objective_type == "reinforce":
            return self.on_policy_rl_loss(batch)

"""The main module for different objectives to train the policy (llm)."""

from itertools import chain
from typing import Any, List, Optional

import torch
from absl import flags
from torch.utils.data import DataLoader

from src.llm import LLM
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


class LossCalculator:

    def __init__(
        self,
        policy_lm: LLM,
        objective_type: str,
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

    def compute_policy_log_probs(self, input_texts: List[str], row_ids: List[str], sample_outputs: List[List[str]]) -> Any:
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
            num_samples_per_example = len(sample_outputs[0])
            expanded_input_texts = list(chain.from_iterable([[text] * num_samples_per_example for text in input_texts]))
            expanded_row_ids = list(chain.from_iterable([[row_id] * num_samples_per_example for row_id in row_ids]))
            flattened_sample_outputs = list(chain.from_iterable(sample_outputs))
            data = self.policy_lm.prepare_text_for_train(
                texts=expanded_input_texts, output_texts=flattened_sample_outputs, row_ids=expanded_row_ids
            )
            dataset = DictDataset(data)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=len(dataset),
                num_workers=1,
            )
            for batch in dataloader:
                sequence_log_probs, token_log_probs, logits, masked_labels = self.policy_lm.train(batch, per_step_scores=True)

            _, sequence_length, vocab_size = logits.size()
            sequence_log_probs = sequence_log_probs.view(batch_size, num_samples_per_example)

            # the sequence length in the token_log_probs has been shifted to the right by 1 position.
            token_log_probs = token_log_probs.view(batch_size, num_samples_per_example, sequence_length - 1)
            logits = logits.view(batch_size, num_samples_per_example, sequence_length, vocab_size)
            masked_labels = masked_labels.view(batch_size, num_samples_per_example, sequence_length)
            return sequence_log_probs, token_log_probs, logits, masked_labels

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
    ) -> torch.Tensor:
        """We have to feed the input along with new sampled outputs to train
        the policy."""
        original_input_texts = batch["texts"]
        original_row_ids = batch["row_ids"]

        sequence_log_probs, token_log_probs, logits, masked_labels = self.compute_policy_log_probs(
            input_texts=original_input_texts, row_ids=original_row_ids, sample_outputs=sample_outputs
        )
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

        # Implement how you can provide entorpy per-step rewards.
        # Implement how you can provide KL penalty with respect to the reference policy.
        # Test the implementations in a task.

        return loss

    def on_policy_rl_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """This is the function to sample from the same policy and train it
        with RL loss."""
        generations, _ = self.policy_lm.generation_pass(batch, num_return_sequences=FLAGS.rl_sample_size)
        cleaned_samples = [text.removeprefix("assistant\n\n") for text in generations]
        templated_samples = [
            self.policy_lm.output_template.format(output=f"Final Answer: {sample}") for sample in cleaned_samples
        ]
        batch_size = len(templated_samples) // FLAGS.rl_sample_size
        sample_outputs = [
            templated_samples[b_idx * FLAGS.rl_sample_size : (b_idx + 1) * FLAGS.rl_sample_size] for b_idx in range(batch_size)
        ]
        # I have to implement the actual reward function.
        sample_rewards = [[1.1, 1.2, 1.3, 1.4], [1.3, 1.4, 2.3, 3.4]]
        loss = self.reinforce_loss(batch, sample_outputs, sample_rewards)
        return loss

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Based on the train objective type, call the right loss function."""

        if self.objective_type == "teacher_forcing":
            return self.teacher_forcing_loss(batch)

        elif self.objective_type == "reinforce":
            return self.on_policy_rl_loss(batch)

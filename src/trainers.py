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

flags.DEFINE_string("objective_type", "reinforce", "Different objectives to get the loss for training the llm.")


class LossCalculator:

    def __init__(
        self,
        policy_lm: LLM,
        value_lm: Optional[LLM] = None,
        ref_policy_lm: Optional[LLM] = None,
        iterative_computation: bool = False,
        reward_normalization_type: str = "zscore",
    ):
        super().__init__()
        self.policy_lm = policy_lm
        self.value_lm = value_lm
        self.ref_policy_lm = ref_policy_lm
        self.iterative_computation = iterative_computation
        self.reward_normalization_type = reward_normalization_type

    def compute_policy_log_probs(self, input_texts: List[str], row_ids: List[str], sample_outputs: List[List[str]]) -> Any:
        """Feed the input along with the sampled output to compute the log
        probability of the policy for these actions."""
        batch_size = len(input_texts)
        if self.iterative_computation:
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
        if self.reward_normalization_type == "zscore":
            return z_scoring(rewards)

        elif self.reward_normalization_type == "normalize":
            return normalize(rewards)

        elif self.reward_normalization_type == "mml_normalize":
            return mml_normalize(rewards)

        elif self.reward_normalization_type == "rloo_normalize":
            return rloo_normalize(rewards)

    def reinforce_style(
        self, batch: torch.utils.data.Dataset, sample_outputs: List[List[str]], sample_output_rewards: List[List[float]]
    ) -> torch.Tensor:
        """We have to feed the input along with new sampled outputs to train
        the policy."""
        original_input_texts = batch["texts"]
        original_row_ids = batch["row_ids"]

        sequence_log_probs, token_log_probs, logits, masked_labels = self.compute_policy_log_probs(
            input_texts=original_input_texts, row_ids=original_row_ids, sample_outputs=sample_outputs
        )
        normalized_rewards = self.normalize_rewards(sample_output_rewards)
        if self.iterative_computation:
            # with iterative computation, we are dealing with a list of tensors.
            # sequence length might be different between examples.
            sequence_log_probs = torch.cat(sequence_log_probs, dim=0)
            batch_size = len(original_input_texts)
            # second dimension is the number of samples per example.
            sequence_log_probs = sequence_log_probs.view(batch_size, -1)

        if self.reward_normalization_type == "mml_normalize":
            loss = -torch.mean(torch.sum(sequence_log_probs * normalized_rewards, dim=1), dim=0)
        else:
            loss = -torch.mean(torch.mean(sequence_log_probs * normalized_rewards, dim=1), dim=0)

        # Implement how you can subtract baseline in the Reinforce along these rewards samplings.
        # Implement how you can provide entorpy per-step rewards.
        # Implement how you can provide KL penalty with respect to the reference policy.
        # Test the implementations in a task.

        return loss

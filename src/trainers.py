"""The main module for different objectives to train the policy (llm)."""

from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from absl import flags, logging

from src.llm import LLM
FLAGS = flags.FLAGS

flags.DEFINE_string("objective_type", "reinforce", "Different objectives to get the loss for training the llm.")

class LossCalculator:

    def __init__(self, policy_lm: LLM, value_lm: Optional[LLM] = None, ref_policy_lm: Optional[LLM] = None):
        super().__init__()
        self.policy_lm = policy_lm
        self.value_lm = value_lm
        self.ref_policy_lm = ref_policy_lm
        
    def reinforce(self, samples: torch.Tensor, sample_rewards: torch.Tensor): 
        
        
    
    
    
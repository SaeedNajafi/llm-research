"""Main binary to run the application to train or test with the model."""

from typing import Any

from absl import app, flags

from src.load_llama import load_peft_model_and_tokenizer, shard_model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from src.model_utils import set_random_seed

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "the seed number")


def main(argv: Any) -> None:
    """Main function to launch the train and inference scripts."""

    del argv

    # set random seed.
    set_random_seed(FLAGS.seed)

    model, tokenizer = load_peft_model_and_tokenizer(use_mp=True, use_fa=True, adapter_name="lora", is_trainable=False)
    sharded_model = shard_model(
                              model,
                              LlamaDecoderLayer,
                              use_mp=True,
                              use_activation_checkpointing=True,
                              strategy="NO_SHARD"
                          )
    

if __name__ == "__main__":
    app.run(main)

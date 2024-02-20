"""Main binary to run the application to train or test with the model."""

from typing import Any

from absl import app, flags

from src.load_llama import load_peft_model_and_tokenizer
from src.model_utils import set_random_seed

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "the seed number")


def main(argv: Any) -> None:
    """Main function to launch the train and inference scripts."""

    del argv

    # set random seed.
    set_random_seed(FLAGS.seed)

    model, tokenizer = load_peft_model_and_tokenizer(use_mp=True, use_fa=False, adapter_name="lora", is_trainable=False)


if __name__ == "__main__":
    app.run(main)

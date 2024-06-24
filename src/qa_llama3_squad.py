import ast
import random
from typing import Any, List, Tuple

import pandas as pd
from absl import app, flags
from torch.utils.data import DataLoader

from src.general_utils import DictDataset, test_loop, train_loop, white_space_fix

# Let's generate paraphrases with Llama3.
from src.llama3 import LlamaQA
from src.metrics import qa_metric
from src.model_utils import set_random_seed
from src.squadv2_llama3_instructions import explanation_icl_input, explanation_instruction, normal_icl_input, normal_instruction

FLAGS = flags.FLAGS

flags.DEFINE_string("output_file", "a name", "the name of file to read data to.")

flags.DEFINE_string("train_file_name", "a name", "the name of train file.")
flags.DEFINE_string("dev_file_name", "a name", "the name of dev file.")
flags.DEFINE_string("test_file_name", "a name", "the name of test file.")

flags.DEFINE_string("experiment_type", "normal_icl", "Normal few-shot learning without explanations.")
flags.DEFINE_string("run_type", "inference", "inference or different types of training.")


train_batch_size = 2
model_path = "/scratch/ssd004/scratch/snajafi/checkpoints/llama3-squadv2.0"


def process_squad_dataset(
    file_name: str, llama3_instruction_llama: str, llama3_input_template: str, llama3_output_template: str
) -> Tuple[List[str], List[str], List[str], List[List[str]]]:
    """Read and pre-process the squadv2 dataset for my application."""

    dataset = pd.read_csv(file_name, sep="\t").to_dict(orient="records")

    if FLAGS.experiment_type == "normal_icl":
        instruction = normal_icl_input

    elif FLAGS.experiment_type == "normal_no_icl":
        instruction = normal_instruction

    elif FLAGS.experiment_type == "explanation_icl":
        instruction = explanation_icl_input

    elif FLAGS.experiment_type == "explanation_no_icl":
        instruction = explanation_instruction

    llama3_instruction = llama3_instruction_llama.format(instruction=instruction)

    next_example_number = 11 if "_no_" not in FLAGS.experiment_type else -1
    squad_inputs = []
    squad_outputs = []
    gold_outputs = []
    squad_ids = []
    idx = 0
    for row in dataset:
        idx += 1
        context = row["context"]
        question = row["question"]
        gold_answers = ast.literal_eval(row["answers"])
        context = white_space_fix(context)
        question = white_space_fix(question)
        if FLAGS.experiment_type == "normal_icl":
            user_final_message = f"Passage_{next_example_number}: {context}"
            user_final_message += f"\nQuestion_{next_example_number}: {question}"
            user_final_message += f"\nFinal Answer_{next_example_number}: "
        elif FLAGS.experiment_type == "normal_no_icl":
            user_final_message = f"Passage: {context}"
            user_final_message += f"\nQuestion: {question}"
            user_final_message += "\nFinal Answer: "
        elif FLAGS.experiment_type == "explanation_icl":
            user_final_message = f"Passage_{next_example_number}: {context}"
            user_final_message += f"\nQuestion_{next_example_number}: {question}"
            user_final_message += f"\nExplanations and Thought Process_{next_example_number}: "
        elif FLAGS.experiment_type == "explanation_no_icl":
            user_final_message = f"Passage: {context}"
            user_final_message += f"\nQuestion: {question}"
            user_final_message += "\nExplanations and Thought Process and Final Answer: "

        llama3_input = llama3_input_template.format(input=user_final_message)
        squad_input = f"{llama3_instruction}{llama3_input}"
        squad_inputs.append(squad_input)
        squad_ids.append(str(idx))

        gold_outputs.append(gold_answers)
        gold_answer = random.choice(gold_answers)

        if FLAGS.experiment_type == "normal_icl":
            squad_output = llama3_output_template.format(output=f"Final Answer_{next_example_number}: {gold_answer}")
            squad_outputs.append(squad_output)
        elif FLAGS.experiment_type == "normal_no_icl":
            squad_output = llama3_output_template.format(output=f"Final Answer: {gold_answer}")
            squad_outputs.append(squad_output)
    return squad_inputs, squad_ids, squad_outputs, gold_outputs


def no_training_inference_main() -> None:

    # Create model.
    set_random_seed(42)

    if FLAGS.experiment_type in ["normal_no_icl", "explanation_no_icl"]:
        model = LlamaQA(
            device="cuda:0", seed=42, lm_top_p=0.9, temperature=0.6, lm_input_max_length=2048 - 1024, lm_output_max_length=1024
        )
        eval_batch_size = 16
    else:
        model = LlamaQA(
            device="cuda:0", seed=42, lm_top_p=0.9, temperature=0.6, lm_input_max_length=8192 - 1024, lm_output_max_length=1024
        )
        eval_batch_size = 4

    model.to_device()

    squad_inputs, squad_ids, _, gold_outputs = process_squad_dataset(
        file_name=FLAGS.test_file_name,
        llama3_instruction_llama=model.llama3_instruction_llama,
        llama3_input_template=model.llama3_input_template,
        llama3_output_template=model.llama3_output_template,
    )
    data = model.prepare_text_for_inference(squad_inputs, row_ids=squad_ids, gold_answers=gold_outputs)
    dataset = DictDataset(data)
    data_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    # Run on the test data.
    test_loop(
        model=model,
        mode="test",
        model_path=f"{model_path}_{FLAGS.experiment_type}",
        prediction_file_name=FLAGS.output_file,
        test_dataloader=data_loader,
        metric=qa_metric,
    )


def train_main() -> None:

    # Create model.
    set_random_seed(42)
    if FLAGS.experiment_type in ["normal_no_icl"]:
        model = LlamaQA(
            device="cuda:0", seed=42, lm_top_p=0.9, temperature=0.6, lm_input_max_length=2048 - 1024, lm_output_max_length=1024
        )
        eval_batch_size = 16
        train_batch_size = 4
    else:
        model = LlamaQA(
            device="cuda:0", seed=42, lm_top_p=0.9, temperature=0.6, lm_input_max_length=8192 - 1024, lm_output_max_length=1024
        )
        eval_batch_size = 4
        train_batch_size = 1

    model.to_device()

    train_squad_inputs, train_squad_ids, train_squad_outputs, train_gold_outputs = process_squad_dataset(
        file_name=FLAGS.train_file_name,
        llama3_instruction_llama=model.llama3_instruction_llama,
        llama3_input_template=model.llama3_input_template,
        llama3_output_template=model.llama3_output_template,
    )

    train_data = model.prepare_text_for_train(train_squad_inputs, train_squad_outputs, train_squad_ids)
    train_dataset = DictDataset(train_data)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_squad_inputs, val_squad_ids, val_squad_outputs, val_gold_outputs = process_squad_dataset(
        file_name=FLAGS.dev_file_name,
        llama3_instruction_llama=model.llama3_instruction_llama,
        llama3_input_template=model.llama3_input_template,
        llama3_output_template=model.llama3_output_template,
    )

    val_data = model.prepare_text_for_inference(val_squad_inputs, val_squad_ids, gold_answers=val_gold_outputs)
    val_dataset = DictDataset(val_data)
    val_data_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    # Run on the train loop
    train_loop(
        model=model,
        mode="train",
        model_path=f"{model_path}_{FLAGS.run_type}_{FLAGS.experiment_type}",
        metric_to_save="sentence_similarity",
        max_epochs=10,
        training_steps=10000000,  # not important
        steps_per_checkpoint=10,
        metric=qa_metric,
        train_dataloader=train_data_loader,
        eval_dataloader=val_data_loader,
    )


def main(argv: Any) -> None:
    del argv
    if FLAGS.run_type == "inference":
        no_training_inference_main()
    elif FLAGS.run_type == "train":
        train_main()


if __name__ == "__main__":
    app.run(main)

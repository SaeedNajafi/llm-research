"""Notebook to send and run the gemma2 predictions.

python gemma2-squad-predictions.py --server_url http://172.17.8.8:47927/v1 \
--model_name /model-weights/gemma-2-9b-it \
--output_path /scratch/ssd004/scratch/snajafi/gemma2-9b-predictions
"""

import csv
import io
import time
from typing import Any

import numpy as np
from absl import app, flags

from src.llm_client import parallel_generator
from src.utils.data_utility import process_squadv2_dataset

FLAGS = flags.FLAGS

# Gemma2 chat templates.
instruction_template = "<bos><start_of_turn>user\n{instruction}"
input_template = "\n{input}<end_of_turn>\n<start_of_turn>model"
output_template = "\n{output} <end_of_turn>"


def main(argv: Any) -> None:
    """Example to use the client."""
    del argv
    input_file = FLAGS.test_file

    # experiment_types = ["normal_no_icl", "explanation_no_icl", "normal_icl", "explanation_icl"]
    experiment_types = ["normal_no_icl"]
    for experiment_type in experiment_types:
        output_file = FLAGS.prediction_file
        # read the input data.
        squad_inputs, squad_ids, _, gold_outputs = process_squadv2_dataset(
            input_file, experiment_type, instruction_template, input_template, output_template
        )
        print(len(squad_inputs))
        print(len(squad_ids))
        print(len(gold_outputs))
        print(squad_inputs[0])
        print(squad_ids[0])
        print(gold_outputs[0])
        start_time = time.perf_counter()
        with io.open(output_file, mode="w", encoding="utf-8") as out_fp:
            writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
            headers = ["potential_answer", "prediction_score", "row_id", "gold_answer"]
            writer.writerow(headers)
            responses = parallel_generator(
                server_url=FLAGS.server_url,
                model_name=FLAGS.model_name,
                inputs=squad_inputs,
                num_threads=4,
                max_new_tokens=256,
                max_retries=3,
                seconds_between_retries=10,
                request_batch_size=1,
                stop_token_ids=[1, 107],
            )
            end_time = time.perf_counter()
            print(f"Finished prediction in {end_time - start_time} seconds!")
            for idx, response in enumerate(responses):
                to_write = [response.text]
                if response.logprobs is not None:
                    to_write.append(np.mean(response.logprobs.token_logprobs))
                else:
                    to_write.append(0.0)
                to_write.append(squad_ids[idx])
                to_write.append(gold_outputs[idx])
                writer.writerow(to_write)

        print(f"Finished i/o in {time.perf_counter() - end_time} seconds!")


if __name__ == "__main__":
    app.run(main)

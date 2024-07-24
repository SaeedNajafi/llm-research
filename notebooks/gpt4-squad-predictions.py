"""Notebook to send and run the gemma2 predictions.

python gpt4-squad-predictions.py \
--prediction_file /scratch/ssd004/scratch/snajafi/gpt4-omini-predictions/original_validation_part1.gtp4-omini.tsv
"""

import csv
import io
import time
from typing import Any

import numpy as np
from absl import app, flags

from src.openai_client import parallel_generator
from src.utils.data_utility import process_squadv2_dataset

FLAGS = flags.FLAGS

# my gpt4-omini chat templates.
instruction_template = "<s> Instruction:\n{instruction} </s>\n"
input_template = "<s> Input:\n{input} </s>\n"
output_template = "<s> Output:\n{output} </s>"


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
        full_messages = [{"role": "user", "content": squad_input} for squad_input in squad_inputs]
        start_time = time.perf_counter()
        with io.open(output_file, mode="w", encoding="utf-8") as out_fp:
            writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
            headers = ["potential_answer", "prediction_score", "row_id", "gold_answer"]
            writer.writerow(headers)

            responses = parallel_generator(
                api_key=FLAGS.openai_key,
                model_name=FLAGS.model_name,
                messages=full_messages,
                num_threads=FLAGS.num_threads,
                max_new_tokens=FLAGS.max_new_tokens,
                max_retries=FLAGS.max_retries,
                seconds_between_retries=FLAGS.seconds_between_retries,
                request_batch_size=FLAGS.request_batch_size,
                stop_tokens=FLAGS.stop_tokens,
            )
            end_time = time.perf_counter()
            print(f"Finished prediction in {end_time - start_time} seconds!")
            for idx, response in enumerate(responses):
                if response.logprobs is not None:
                    logprobs = [log_content.logprob for log_content in response.logprobs.content if log_content.logprob != 0]
                    prediction_score = np.mean(logprobs) if len(logprobs) > 0 else 0.0
                else:
                    prediction_score = -1.0
                to_write = [response.message.content, prediction_score, squad_ids[idx], gold_outputs[idx]]
                writer.writerow(to_write)

        print(f"Finished i/o in {time.perf_counter() - end_time} seconds!")


if __name__ == "__main__":
    app.run(main)

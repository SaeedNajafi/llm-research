# Notebook to send and run the gemma2 predictions.
import csv
import io
import time

import numpy as np

from src.llm_client import parallel_generator
from src.utils.data_utility import process_squadv2_dataset

# Gemma2 chat templates.
instruction_template = "<bos><start_of_turn>user\n{instruction}"
input_template = "\n{input}<end_of_turn>\n<start_of_turn>model"
output_template = "\n{output} <end_of_turn>"

# Model name and server address
model_name = "/home/saeednjf/nearline/rrg-afyshe/saeednjf/checkpoints/squadv2/gemma2/0.1_13_lora_rank_64/final_model"
server_address = "http://172.17.8.9:58023/v1"
output_path = "/home/saeednjf/nearline/rrg-afyshe/saeednjf/checkpoints/squadv2/gemma2/0.1_13_lora_rank_64/final-predictions"


input_file = "../data/0.1-shot-datasets/squad/original_validation.tsv"


experiment_types = ["normal_no_icl"]
for experiment_type in experiment_types:
    output_file = f"squadv2_predictions_original_validation.{experiment_type}.csv"
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
    with io.open(f"{output_path}/{output_file}", mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
        headers = ["potential_answer", "prediction_score", "row_id", "gold_answer"]
        writer.writerow(headers)
        responses = parallel_generator(
            server_url=server_address,
            model_name=model_name,
            inputs=squad_inputs,
            num_threads=1,
            max_new_tokens=256,
            max_retries=3,
            seconds_between_retries=5,
            request_batch_size=32,
            stop_token_ids=[1, 107],
        )
        end_time = time.perf_counter()
        print(f"Finished prediction in {end_time - start_time} seconds!")
        for idx, response in enumerate(responses):
            to_write = [response.text, np.mean(response.logprobs.token_logprobs), squad_ids[idx], gold_outputs[idx]]
            writer.writerow(to_write)

    print(f"Finished i/o in {time.perf_counter() - end_time} seconds!")

# Notebook to send and run the llama3 predictions.
import csv
import io
import time

import numpy as np

from src.llm_client import parallel_generator
from src.utils.data_utility import process_squadv2_dataset

# In[4]:


# Llama3 chat templates.
instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction} <|eot_id|>"
input_template = "<|start_header_id|>user<|end_header_id|>\n\n{input} <|eot_id|>"
output_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{output} <|eot_id|>"


# In[9]:


# Model name and server address
model_name = "/model-weights/Meta-Llama-3-8B-Instruct"
server_address = "http://172.17.8.9:58023/v1"
output_path = "/scratch/ssd004/scratch/snajafi/checkpoints/llama3-predictions"


input_file = "../data/0.1-shot-datasets/squad/original_validation.tsv"


experiment_types = ["normal_no_icl", "explanation_no_icl", "normal_icl", "explanation_icl"]
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
            num_threads=4,
            max_new_tokens=256,
            max_retries=3,
            seconds_between_retries=5,
            request_batch_size=16,
        )
        end_time = time.perf_counter()
        print(f"Finished prediction in {end_time - start_time} seconds!")
        for idx, response in enumerate(responses):
            to_write = [response.text, np.mean(response.logprobs.token_logprobs), squad_ids[idx], gold_outputs[idx]]
            writer.writerow(to_write)

    print(f"Finished i/o in {time.perf_counter() - end_time} seconds!")

#!/bin/bash

set -x

gpt4o_path="/scratch/ssd004/scratch/snajafi/final-gpt4-omini-squadv2"
llama3_path="/scratch/ssd004/scratch/snajafi/llama3.1-8b-predictions"
gemma2_path="/scratch/ssd004/scratch/snajafi/gemma2-9b-predictions"

'''
python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${llama3_path}/squadv2_original_validation_normal_no_icl.llama3.1-8b.tsv > ${llama3_path}/squadv2_original_validation_normal_no_icl.llama3.1-8b.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${llama3_path}/squadv2_original_validation_normal_with_icl.llama3.1-8b.tsv > ${llama3_path}/squadv2_original_validation_normal_with_icl.llama3.1-8b.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${llama3_path}/squadv2_original_validation_explanation_no_icl.llama3.1-8b.tsv > ${llama3_path}/squadv2_original_validation_explanation_no_icl.llama3.1-8b.metrics.txt 2>&1

 python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${llama3_path}/squadv2_original_validation_explanation_with_icl.llama3.1-8b.tsv > ${llama3_path}/squadv2_original_validation_explanation_with_icl.llama3.1-8b.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${gpt4o_path}/squadv2_original_validation_normal_no_icl.gtp4-omini.tsv > ${gpt4o_path}/squadv2_original_validation_normal_no_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${gpt4o_path}/squadv2_original_validation_normal_with_icl.gtp4-omini.tsv > ${gpt4o_path}/squadv2_original_validation_normal_with_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${gpt4o_path}/squadv2_original_validation_explanation_no_icl.gtp4-omini.tsv > ${gpt4o_path}/squadv2_original_validation_explanation_no_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
--input_file ${gpt4o_path}/squadv2_original_validation_explanation_with_icl.gtp4-omini.tsv > ${gpt4o_path}/squadv2_original_validation_explanation_with_icl.gtp4-omini.metrics.txt 2>&1

 python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
 --input_file ${gemma2_path}/squadv2_original_validation_normal_no_icl.gemma2-9b.tsv > ${gemma2_path}/squadv2_original_validation_normal_no_icl.gemma2-9b.metrics.txt 2>&1
 '''

python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
 --input_file ${gemma2_path}/squadv2_original_validation_normal_with_icl.gemma2-9b.tsv > ${gemma2_path}/squadv2_original_validation_normal_with_icl.gemma2-9b.metrics.txt 2>&1

 python src/metrics.py --metric_device=cuda:0 --metric_type llm2vec \
 --input_file ${gemma2_path}/squadv2_original_validation_explanation_no_icl.gemma2-9b.tsv > ${gemma2_path}/squadv2_original_validation_explanation_no_icl.gemma2-9b.metrics.txt 2>&1

#!/bin/bash

set -x

# path="/h/snajafi/final-gpt4-omini-squadv2"
# path="/scratch/ssd004/scratch/snajafi/llama3.1-8b-predictions"
path="/scratch/ssd004/scratch/snajafi/gemma2-9b-predictions"

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_normal_no_icl.gemma2-9b.tsv > ${path}/squadv2_original_validation_normal_no_icl.gemma2-9b.metrics.txt 2>&1

'''
python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_normal_no_icl.llama3.1-8b.tsv > ${path}/squadv2_original_validation_normal_no_icl.llama3.1-8b.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_explanation_no_icl.llama3.1-8b.tsv > ${path}/squadv2_original_validation_explanation_no_icl.llama3.1-8b.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_normal_no_icl.gtp4-omini.tsv > ${path}/squadv2_original_validation_normal_no_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_normal_with_icl.gtp4-omini.tsv > ${path}/squadv2_original_validation_normal_with_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_explanation_no_icl.gtp4-omini.tsv > ${path}/squadv2_original_validation_explanation_no_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_explanation_with_icl.gtp4-omini.tsv > ${path}/squadv2_original_validation_explanation_with_icl.gtp4-omini.metrics.txt 2>&1
'''

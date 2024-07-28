#!/bin/bash

set -x

path="/h/snajafi/final-gpt4-omini-squadv2"

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_normal_no_icl.gtp4-omini.tsv > ${path}/squadv2_original_validation_normal_no_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_normal_with_icl.gtp4-omini.tsv > ${path}/squadv2_original_validation_normal_with_icl.gtp4-omini.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_original_validation_explanation_no_icl.gtp4-omini.tsv > ${path}/squadv2_original_validation_explanation_no_icl.gtp4-omini.metrics.txt 2>&1

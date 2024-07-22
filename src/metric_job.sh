#!/bin/bash

set -x

path="/scratch/ssd004/scratch/snajafi/gemma2-9b-predictions"

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_predictions_original_validation.explanation_icl.csv >> ${path}/explanation_icl.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_predictions_original_validation.normal_icl.csv >> ${path}/normal_icl.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_predictions_original_validation.normal_no_icl.csv >> ${path}/normal_no_icl.metrics.txt 2>&1

python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file ${path}/squadv2_predictions_original_validation.explanation_no_icl.csv >> ${path}/explanation_no_icl.metrics.txt 2>&1

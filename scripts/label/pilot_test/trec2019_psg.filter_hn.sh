#!/bin/bash
experiment_root=experiments/pilot_test/trec2019 #TODO: custom path
qrels=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path

rerank_experiment=wo_qe #TODO: custom path
hard_negatives=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
output=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/filtered_hn.jsonl #TODO: custom path

python -m preprocessing.hard_negatives.filter_hard_negatives_for_train \
--hn ${hard_negatives} --hn_topk 100 --qrels ${qrels} --output ${output}

# experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/filtered_hn.jsonl
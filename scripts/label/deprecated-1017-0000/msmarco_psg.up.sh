#!/bin/bash
# up_score_threshold=$1 #TODO: input arg
# ranking_jsonl=$2 #TODO: input arg
# output=$3 #TODO: input arg

up_score_threshold=28 #TODO: custom arg
ranking_jsonl=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
output=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/pseudo_positives.jsonl #TODO: custom path

python -m preprocessing.pseudo_labeling.filter_pseudo_positives \
--thr ${up_score_threshold} --ranking_jsonl ${ranking_jsonl} --output ${output}
"""
#> ranking_jsonl:       experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl
#> output       :       experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/pseudo_positives.jsonl
#> topk -1, thr 28.0
#> The # of positives: Min 0, Max 439, Mean 5.69, Median 1.0
"""


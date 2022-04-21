#!/bin/bash
# up_score_threshold=$1 #TODO: input arg
# ranking_jsonl=$2 #TODO: input arg
# output=$3 #TODO: input arg

thr=-1.0 #TODO: custom arg
topk=3 #TODO: custom arg
ranking_jsonl=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/ranking.jsonl #TODO: custom path
# output=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/pseudo_positives.topk3.jsonl #TODO: custom path
output=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/pseudo_qrels.topk3.tsv #TODO: custom path

python -m preprocessing.pseudo_labeling.filter_pseudo_positives \
--labeled_qrels data/qrels.train.tsv \
--thr ${thr} --topk ${topk} --ranking_jsonl ${ranking_jsonl} --output ${output}
# [Oct 29, 02:07:30] #> Loading qrels from data/qrels.train.tsv ...
# [Oct 29, 02:07:32] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.

# [Oct 29, 02:07:33] #> ranking_jsonl:    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/ranking.jsonl
# [Oct 29, 02:07:33] #> output       :    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/pseudo_qrels.topk3.tsv
# [Oct 29, 02:07:33] #> topk 3, thr -1.0
# [Oct 29, 02:13:26] #> The # of positives: Min 3, Max 3, Mean 3.00, Median 3.0

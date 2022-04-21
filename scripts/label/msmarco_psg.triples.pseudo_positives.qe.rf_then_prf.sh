#!/bin/bash
# hard_negatives=$1 #TODO: input arg
# output=$2 #TODO: input arg

hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
output=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/triples.train.small.ids.hn.jsonl #TODO: custom path

# qrels=data/qrels.train.tsv
qrels=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/pseudo_qrels.topk3.tsv #TODO: custom path

python -m preprocessing.hard_negatives.construct_new_train_triples \
--hn_topk 100 --n_triples 40000000 --qrels ${qrels} --n_negatives 1 \
--hn ${hard_negatives} --output ${output}

# #> Load positives
# [Oct 29, 02:24:46] #> Loading qrels from experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/pseudo_qrels.topk3.tsv ...
# [Oct 29, 02:24:50] #> Loaded qrels for 502939 unique queries with 4.06 positives per query on average.

# #> Elapsed time for loading positives: 5.145989179611206

# #> Gather positive pairs
# #> Elapsed time for gathering positive pairs: 1.1186130046844482

# #> Load negatives
# #> Load experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl
#         pid for 121352 => 5748989
#         pid for 634306 => 7926774
#         pid for 920825 => 3285652
#         pid for 510633 => 1050366
#         pid for 737889 => 3523789
#         pid for 674172 => 710286
#         pid for 303205 => 8394396
#         pid for 570009 => 578030
#         pid for 492875 => 2533238
#         pid for 54528 => 7416534
# #> Elapsed time for loading negatives: 257.92709589004517

# #> The # of positives: Min 4, Max 10, Mean 4.06, Median 4.0
# #> The # of negatives: Min 90, Max 100, Mean 96.12, Median 96.0


# #> Save [qid, ppid, npid#1, npid#2, ..., npid#N] to experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/triples.train.small.ids.hn.jsonl (# of samples = 40000000)
# #> Elapsed time for saving triples: 189.53262543678284


# #> output:               experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/triples.train.small.ids.hn.jsonl

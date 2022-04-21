#!/bin/bash
# hard_negatives=$1 #TODO: input arg
# output=$2 #TODO: input arg

hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
output=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/filtered_hn.jsonl #TODO: custom path

python -m preprocessing.hard_negatives.filter_hard_negatives_for_train \
--hn ${hard_negatives} --hn_topk 100 --qrels data/qrels.train.tsv --output ${output}

# [Oct 17, 21:57:48] #> Loading qrels from data/qrels.train.tsv ...
# [Oct 17, 21:57:50] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.

# #> Elapsed time for loading positives: 2.525296688079834
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
# #> Elapsed time for loading negatives: 244.80749821662903

# #> The # of positives: Min 1, Max 7, Mean 1.06, Median 1.0
# #> The # of negatives: Min 93, Max 100, Mean 99.04, Median 99.0

# #> Save filtered hard negatives to experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/filtered_hn.jsonl
# #> Elapsed time for constructing triples: 9.203495264053345
# #> output:               experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/filtered_hn.jsonl

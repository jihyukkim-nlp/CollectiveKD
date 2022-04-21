#!/bin/bash
# hard_negatives=$1 #TODO: input arg
# output=$2 #TODO: input arg

hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
output=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl #TODO: custom path

python -m preprocessing.utils.construct_new_train_triples \
--hn ${hard_negatives} --output ${output}

# [Oct 08, 10:17:26] #> Loading qrels from data/qrels.train.tsv ...
# [Oct 08, 10:17:29] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.

# #> Elapsed time for loading positives: 6.399592161178589
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
# #> Elapsed time for loading negatives: 270.35175943374634

# #> The # of positives: Min 1, Max 7, Mean 1.06, Median 1.0
# #> The # of negatives: Min 93, Max 100, Mean 99.04, Median 99.0

# #> Construct new train triples: list of (query ID, positive psg ID, negative psg ID)
# #> Elapsed time for constructing triples: 14.668428659439087
# #> Shuffle
# #> Elapsed time for shuffling triples: 50.56377410888672
# #> Only retain top-40000000 triples to save disk (training often be early terminated before using all train triples).
# #> Save triples to experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl
# #> Elapsed time for saving triples: 59.9421021938324
# #> output:               experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl

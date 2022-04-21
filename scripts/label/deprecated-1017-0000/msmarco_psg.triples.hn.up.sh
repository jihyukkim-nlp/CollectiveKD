#!/bin/bash
# hard_negatives=$1 #TODO: input arg
# pseudo_positives=$2 #TODO: input arg
# output=$3 #TODO: input arg

hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
pseudo_positives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/pseudo_positives.jsonl #TODO: custom path
output=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.train.small.ids.jsonl #TODO: custom path

python -m preprocessing.utils.construct_new_train_triples \
--hn ${hard_negatives} --pp ${pseudo_positives} --output ${output}
# #> Load experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/pseudo_positives.jsonl
# #> The # of positives: Min 0, Max 50, Mean 4.97, Median 1.0
# [Oct 07, 13:07:58] #> Loading qrels from data/qrels.train.tsv ...
# [Oct 07, 13:08:00] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.

# #> Elapsed time for loading positives: 7.567443132400513
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
# #> Elapsed time for loading negatives: 227.8824281692505

# #> The # of positives: Min 1, Max 54, Mean 5.69, Median 2.0
# #> The # of negatives: Min 47, Max 100, Mean 94.41, Median 98.0

# #> Construct new train triples: list of (query ID, positive psg ID, negative psg ID)
# #> Elapsed time for constructing triples: 27.76762294769287
# #> Shuffle 
# #> Elapsed time for shuffling triples: 154.3175172805786
# #> Only retain top-40000000 triples to save disk (training often be early terminated before using all train triples).
# #> Save triples to experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.train.small.ids.jsonl
# #> Elapsed time for saving triples: 45.232131481170654
# #> output:               experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.train.small.ids.jsonl
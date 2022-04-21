#!/bin/bash
# hard_negatives=$1 #TODO: input arg
# pseudo_positives=$2 #TODO: input arg
# output=$3 #TODO: input arg

hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
pseudo_positives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/pseudo_positives.jsonl #TODO: custom path
output=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/triples.train.small.ids.jsonl #TODO: custom path

python -m preprocessing.utils.construct_new_train_triples \
--hn ${hard_negatives} --pp ${pseudo_positives} --output ${output}
# #> Load experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/pseudo_positives.jsonl
# #> The # of positives: Min 0, Max 50, Mean 6.02, Median 1.0
# [Oct 07, 13:21:29] #> Loading qrels from data/qrels.train.tsv ...
# [Oct 07, 13:21:32] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.

# #> Elapsed time for loading positives: 7.8523571491241455
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
# #> Elapsed time for loading negatives: 230.46876549720764

# #> The # of positives: Min 1, Max 54, Mean 6.57, Median 2.0
# #> The # of negatives: Min 47, Max 100, Mean 93.53, Median 98.0

# #> Construct new train triples: list of (query ID, positive psg ID, negative psg ID)
# #> Elapsed time for constructing triples: 30.289418935775757
# #> Shuffle
# #> Elapsed time for shuffling triples: 168.19488143920898
# #> Only retain top-40000000 triples to save disk (training often be early terminated before using all train triples).
# #> Save triples to experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/triples.train.small.ids.jsonl
# #> Elapsed time for saving triples: 53.70620012283325
# #> output:               experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/triples.train.small.ids.jsonl
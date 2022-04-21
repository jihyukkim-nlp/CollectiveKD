#!/bin/bash
hard_negatives=$1 #TODO: input arg: e.g., experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl
n_negatives=$2 #TODO: input arg: e.g., 1

[ ! -f "${hard_negatives}" ] && echo "${hard_negatives} does not exist." && return

output=$(dirname ${hard_negatives})/triples.hn${n_negatives}.jsonl

# #TODO: ###########################################################################
# #?@ debugging
# echo ${output}
# echo "(return) check this code: \t\"$(cd $(dirname $0) && pwd)/msmarco_psg.triples.hn.sh\"" && return
# #?@ debugging
# #TODO: ###########################################################################

echo;echo;echo
echo "Construct new train triples using hard negatives (this will take about 8-10 minutes)"
echo;echo;echo

python -m preprocessing.hard_negatives.construct_new_train_triples \
--hn_topk 100 --n_triples 40000000 --qrels data/qrels.train.tsv --n_negatives ${n_negatives} \
--hn ${hard_negatives} --output ${output}

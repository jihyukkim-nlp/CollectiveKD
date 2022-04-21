#!/bin/bash
trec=$1 #TODO: input arg: 2019 or 2020
[ "${trec}" != "2019" ] && [ "${trec}" != "2020" ] && echo "Invalid argument for ``trec``: Enter 2019 or 2020" && return
qrels=data/pilot_test/label/${trec}qrels-pass.train.tsv # labeled positives

# # Top-K passages ranked by ColBERT (student)
# ranking=experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/ranking.jsonl #TODO: custom path
# output=experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/filtered_hn.jsonl #TODO: custom path
# qrels=data/pilot_test/label/2019qrels-pass.train.tsv
# python -m preprocessing.hard_negatives.filter_hard_negatives_for_train --hn ${ranking} --hn_topk 100 --qrels ${qrels} --output ${output}
# output: experiments/pilot_test/trec2019/wo_qe/label.py/2021-10-07_08.30.22/filtered_hn.jsonl 

# Top-K passages ranked by ColBERT-RF (teacher)
ranking_basedir=experiments/pilot_test/trec${trec}/kmeans.k10.beta1.0.clusters10/label.py
ranking=${ranking_basedir}/$(ls ${ranking_basedir})/ranking.jsonl #TODO: custom path
[ -f ${ranking} ] && echo "ranking (${ranking}) does not exist"
output=${ranking_basedir}/$(ls ${ranking_basedir})/filtered_hn.jsonl #TODO: custom path
python -m preprocessing.hard_negatives.filter_hard_negatives_for_train --hn ${ranking} --hn_topk 100 --qrels ${qrels} --output ${output}
# output: experiments/pilot_test/trec2019/kmeans.k10.beta1.0.clusters10/label.py/2021-10-15_19.23.17/filtered_hn.jsonl
# output: experiments/pilot_test/trec2020/kmeans.k10.beta1.0.clusters10/label.py/2021-10-16_17.53.51/filtered_hn.jsonl

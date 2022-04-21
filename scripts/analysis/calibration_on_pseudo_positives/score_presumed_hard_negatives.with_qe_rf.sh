#!/bin/bash
device=$1 #TODO: input arg: GPU device, used as ``CUDA_VISIBLE_DEVICES=${device}``
trec=$2 #TODO: input arg: 2019 or 2020
exp=$3 #TODO: input arg: e.g., experiments/pilot_test/trec2019/kmeans.k10.beta1.0.clusters10/
output=$4 #TODO: input arg: e.g., analysis/calibration_on_pseudo_positives/trec2019/hn_score.with_qe_rf.jsonl

if [ -f "/workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv" ];then
    collection="/workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv"
else
    collection="/workspace/DataCenter/MSMARCO/collection.tsv"
fi
[ ! -f ${collection} ] && echo "collection (${collection}) does not exist"

#! This must be run by as-is (i.e., "./score_presumed~.sh"), instead of "sh ./score_presumed~.sh"
if [[ "${trec}" == "2019" ]];then
    experiment_root=experiments/pilot_test/trec2019 #TODO: custom path
    queries=data/queries.trec2019.tsv #TODO: custom path
elif [[ "${trec}" == "2020" ]];then
    experiment_root=experiments/pilot_test/trec2020 #TODO: custom path
    queries=data/queries.trec2020.tsv #TODO: custom path
else
    echo "Invalid argument for ``trec``: Enter 2019 or 2020" && return
fi

presumed_hn=${experiment_root}/kmeans.k10.beta1.0.clusters10/label.py/$(ls ${experiment_root}/kmeans.k10.beta1.0.clusters10/label.py)/filtered_hn.jsonl #TODO: custom path
[ ! -f ${presumed_hn} ] && echo "presumed_hn (${presumed_hn}) does not exist"
expansion_pt=${exp}/label.py/$(ls ${exp}/label.py)/expansion.pt  

CUDA_VISIBLE_DEVICES=${device} python -m analysis.compute_soft_label_on_topk_passages \
--collection ${collection} --presumed_hn ${presumed_hn} --queries ${queries} --expansion_pt ${expansion_pt} \
--output ${output}
# --output analysis/calibration_on_pseudo_positives/trec${trec}/hn_score.with_qe_rfprf.jsonl
# --output analysis/calibration_on_pseudo_positives/trec${trec}/hn_score.with_qe_prf.jsonl
# --output analysis/calibration_on_pseudo_positives/trec${trec}/hn_score.with_qe_rf.jsonl
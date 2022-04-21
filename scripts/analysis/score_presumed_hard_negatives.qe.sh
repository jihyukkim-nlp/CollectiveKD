#!/bin/bash
device=$1 #TODO: input arg: GPU device, used as ``CUDA_VISIBLE_DEVICES=${device}``
trec=$2 #TODO: input arg: 2019 or 2020
exp=$3 #TODO: input arg: e.g., experiments/pilot_test/trec2019/kmeans.k10.beta1.0.clusters10/
output=$4 #TODO: input arg: e.g., analysis/calibration_on_presumed_negatives/trec2019/hn_score.rf.k10.beta1.0.clusters10.jsonl



if [ -f "/workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv" ];then
    collection="/workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv"
else
    collection="/workspace/DataCenter/MSMARCO/collection.tsv"
fi
[ ! -f ${collection} ] && echo "collection (${collection}) does not exist" && return

#! This must be run by ``source`` (e.g., "source ./score_presumed~.sh" or "./score_presumed~.sh"), instead of ``sh`` (e.g., "sh ./score_presumed~.sh")
[[ "${trec}" != "2019" ]] && [[ "${trec}" != "2020" ]] && echo "Invalid argument for ``trec``: Enter 2019 or 2020" && return
experiment_root=experiments/pilot_test/trec${trec} #TODO: custom path
queries=data/queries.trec${trec}.tsv #TODO: custom path
labeled_positives=data/pilot_test/label/${trec}qrels-pass.train.tsv #TODO: custom path
[ ! -d ${experiment_root} ] && echo "experiment_root (${experiment_root}) does not exist" && return
[ ! -f ${queries} ] && echo "queries (${queries}) does not exist" && return
[ ! -f ${labeled_positives} ] && echo "labeled_positives (${labeled_positives}) does not exist" && return

rerank_experiment=wo_qe #TODO: custom path
presumed_hn=${experiment_root}/wo_qe/label.py/$(ls ${experiment_root}/wo_qe/label.py)/filtered_hn.jsonl #TODO: custom path
[ ! -f ${presumed_hn} ] && echo "presumed_hn (${presumed_hn}) does not exist" && return
expansion_pt=${exp}/label.py/$(ls ${exp}/label.py)/expansion.pt  
[ ! -f ${expansion_pt} ] && echo "expansion_pt (${expansion_pt}) does not exist" && return

CUDA_VISIBLE_DEVICES=${device} python -m analysis.compute_soft_label_on_topk_passages \
--collection ${collection} --presumed_hn ${presumed_hn} --queries ${queries} --expansion_pt ${expansion_pt} --labeled_positives ${labeled_positives} \
--output ${output}
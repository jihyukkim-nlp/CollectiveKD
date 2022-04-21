#!/bin/bash
trec=$1 #TODO: input arg: 2019 or 2020

#! This must be run by as-is (i.e., "./score_presumed~.sh"), instead of "sh ./score_presumed~.sh"
if [[ "${trec}" == "2019" ]];then
    experiment_root=experiments/pilot_test/trec2019 #TODO: custom path
    labeled_qrels=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path
    unlabeled_qrels=data/pilot_test/label/2019qrels-pass.test.tsv #TODO: custom path
    qrels=data/trec2019/2019qrels-pass.txt #TODO: custom path
elif [[ "${trec}" == "2020" ]];then
    experiment_root=experiments/pilot_test/trec2020 #TODO: custom path
    labeled_qrels=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
    unlabeled_qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
    qrels=data/trec2020/2020qrels-pass.txt #TODO: custom path
else
    echo "Invalid argument for ``trec``: Enter 2019 or 2020" && return
fi

presumed_hn=${experiment_root}/kmeans.k10.beta1.0.clusters10/label.py/$(ls ${experiment_root}/kmeans.k10.beta1.0.clusters10/label.py)/filtered_hn.jsonl #TODO: custom path
[ ! -f ${presumed_hn} ] && echo "presumed_hn (${presumed_hn}) does not exist" && return

python -m analysis.compare_scores_on_presumed_negatives --topk 100 \
--presumed_hn ${presumed_hn} --labeled_qrels ${labeled_qrels} --unlabeled_qrels ${unlabeled_qrels} --qrels ${qrels} \
--score_path_list analysis/calibration_on_pseudo_positives/trec${trec}/hn_score.with_qe_prf.jsonl analysis/calibration_on_pseudo_positives/trec${trec}/hn_score.with_qe_rf.jsonl analysis/calibration_on_pseudo_positives/trec${trec}/hn_score.without_qe.jsonl \
--score_model_list 'ColBERT-PRF (teacher)' 'ColBERT-RF (teacher)' 'ColBERT (student)' \
--output analysis/calibration_on_pseudo_positives/trec${trec}/scores_on_presumed_negatives.pkl

#!/bin/bash
trec=$1 #TODO: input arg: 2019 or 2020

#! This must be run by ``source`` (e.g., "source ./score_presumed~.sh" or "./score_presumed~.sh"), instead of ``sh`` (e.g., "sh ./score_presumed~.sh")
[[ "${trec}" != "2019" ]] && [[ "${trec}" != "2020" ]] && echo "Invalid argument for ``trec``: Enter 2019 or 2020" && return
experiment_root=experiments/pilot_test/trec${trec} #TODO: custom path
[ ! -d ${experiment_root} ] && echo "experiment_root (${experiment_root}) does not exist" && return
labeled_qrels=data/pilot_test/label/${trec}qrels-pass.train.tsv #TODO: custom path
unlabeled_qrels=data/pilot_test/label/${trec}qrels-pass.test.tsv #TODO: custom path
qrels=data/trec${trec}/${trec}qrels-pass.txt #TODO: custom path
[ ! -f ${labeled_qrels} ] && echo "labeled_qrels (${labeled_qrels}) does not exist" && return
[ ! -f ${unlabeled_qrels} ] && echo "unlabeled_qrels (${unlabeled_qrels}) does not exist" && return
[ ! -f ${qrels} ] && echo "qrels (${qrels}) does not exist" && return

python -m analysis.construct_scores_to_dataframe \
--labeled_qrels ${labeled_qrels} --unlabeled_qrels ${unlabeled_qrels} --qrels ${qrels} \
\
--score_path_list \
analysis/calibration_on_trec_passages/trec${trec}/score.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24.tsv \
analysis/calibration_on_trec_passages/trec${trec}/score.prf.docs3.k10.beta0.5.clusters24.tsv \
analysis/calibration_on_trec_passages/trec${trec}/score.prf.docs3.k10.beta1.0.clusters24.tsv \
analysis/calibration_on_trec_passages/trec${trec}/score.rf.k10.beta0.5.clusters10.tsv \
analysis/calibration_on_trec_passages/trec${trec}/score.rf.k10.beta1.0.clusters10.tsv \
analysis/calibration_on_trec_passages/trec${trec}/score.colbert.tsv \
\
--score_model_list \
'ColBERT-RF-then-PRF (k 10, beta 0.5) (docs 3, k 10, beta 0.5)' \
'ColBERT-PRF (docs 3, k 10, beta 0.5)' \
'ColBERT-PRF (docs 3, k 10, beta 1.0)' \
'ColBERT-RF (k 10, beta 0.5)' \
'ColBERT-RF (k 10, beta 1.0)' \
'ColBERT' \
--output analysis/calibration_on_trec_passages/trec${trec}/scores_on_trec_passages.pkl

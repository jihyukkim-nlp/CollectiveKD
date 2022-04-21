#!/bin/bash
topk=$1 #TODO: custom arg
thr=$2 #TODO: input arg
# topk=3 #TODO: custom arg
# thr=40 #TODO: custom arg

ranking_jsonl=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/ranking.jsonl #TODO: custom path
[ ! -f ${ranking_jsonl} ] && echo "${ranking_jsonl} does not exist" && return

if [ "${thr}" -eq -1 ];then
    pseudo_qrels=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk${topk}.tsv #TODO: custom path
else
    if [ "${topk}" -eq -1 ];then
        pseudo_qrels=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.thr${thr}.tsv #TODO: custom path
    else
        pseudo_qrels=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk${topk}.thr${thr}.tsv #TODO: custom path
    fi
fi
if [ ! -f ${pseudo_qrels} ];then
    python -m preprocessing.pseudo_labeling.filter_pseudo_positives \
    --labeled_qrels data/qrels.train.tsv \
    --thr ${thr} --topk ${topk} --ranking_jsonl ${ranking_jsonl} --output ${pseudo_qrels}
else
    echo "We have ${pseudo_qrels}"
fi



hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
[ ! -f ${hard_negatives} ] && echo "${hard_negatives} does not exist!" && return

if [ "${thr}" -eq -1 ];then
    triples=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk${topk}.jsonl #TODO: custom path
else
    if [ "${topk}" -eq -1 ];then
        triples=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-thr${thr}.jsonl #TODO: custom path
    else
        triples=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk${topk}-thr${thr}.jsonl #TODO: custom path
    fi
fi

# qrels=data/qrels.train.tsv
qrels=${pseudo_qrels}
[ ! -f ${qrels} ] && echo "${qrels} does not exist" && return

if [ ! -f ${triples} ];then
    python -m preprocessing.hard_negatives.construct_new_train_triples \
    --hn_topk 100 --n_triples 40000000 --qrels ${qrels} --n_negatives 1 \
    --hn ${hard_negatives} --output ${triples}
else
    echo "We have ${triples}"
fi

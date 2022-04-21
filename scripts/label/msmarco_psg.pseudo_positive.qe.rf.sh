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

# [Oct 30, 02:14:42] #> Loading qrels from data/qrels.train.tsv ...
# [Oct 30, 02:14:44] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.
# [Oct 30, 02:14:45] #> ranking_jsonl:    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/ranking.jsonl

# [Oct 30, 02:14:45] #> topk 3, thr 40.0
# [Oct 30, 02:14:45] #> output       :    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv
# [Oct 30, 02:20:25] #> The # of positives: Min 0, Max 3, Mean 1.16, Median 0.0

# [Oct 30, 02:15:11] #> topk 3, thr 41.0
# [Oct 30, 02:15:11] #> output       :    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr41.tsv
# [Oct 30, 02:20:28] #> The # of positives: Min 0, Max 3, Mean 0.88, Median 0.0

# [Oct 30, 02:15:22] #> topk 3, thr 42.0
# [Oct 30, 02:15:22] #> output       :    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr42.tsv
# [Oct 30, 02:20:45] #> The # of positives: Min 0, Max 3, Mean 0.64, Median 0.0

hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl #TODO: custom path
triples=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.train.small.ids.hn.pp-topk${topk}-thr${thr}.jsonl #TODO: custom path

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

# [Oct 30, 02:39:39] #> Loading qrels from experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv ...
# [Oct 30, 02:39:42] #> Loaded qrels for 502939 unique queries with 2.22 positives per query on average.
#> output:               experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.train.small.ids.hn.pp-topk3-thr40.jsonl

# [Oct 30, 02:39:30] #> Loading qrels from experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr41.tsv ...
# [Oct 30, 02:39:32] #> Loaded qrels for 502939 unique queries with 1.94 positives per query on average.
#> output:               experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.train.small.ids.hn.pp-topk3-thr41.jsonl

# [Oct 30, 02:39:18] #> Loading qrels from experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr42.tsv ...
# [Oct 30, 02:39:20] #> Loaded qrels for 502939 unique queries with 1.7 positives per query on average.
# > output:               experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.train.small.ids.hn.pp-topk3-thr42.jsonl


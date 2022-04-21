

# Pseudo-positive V1
#TODO: remove LP from HN
topk=3
thr=40
sh scripts/pseudo_positive_pretraining/msmarco_psg.label_pseudo_positives.rf.sh ${topk} ${thr}
[Nov 03, 12:42:11] #> Loading qrels from data/qrels.train.tsv ...
[Nov 03, 12:42:13] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.
[Nov 03, 12:42:14] #> ranking_jsonl:    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/ranking.jsonl
[Nov 03, 12:42:14] #> output       :    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv
[Nov 03, 12:42:14] #> topk 3, thr 40.0
[Nov 03, 12:48:17] #> The # of positives: Min 0, Max 3, Mean 1.16, Median 0.0
[Nov 03, 12:48:17] #> "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv" contains only pseudo-positives.
output (qrels): experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv
output (triples): experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk3-thr40.jsonl
# 
# Pseudo-positive V2
#TODO: remove LP from HN
topk=3
sh scripts/pseudo_positive_pretraining/msmarco_psg.label_pseudo_positives.rf.sh ${topk} -1
[Nov 03, 12:42:31] #> Loading qrels from data/qrels.train.tsv ...
[Nov 03, 12:42:33] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.
[Nov 03, 12:42:34] #> ranking_jsonl:    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/ranking.jsonl
[Nov 03, 12:42:34] #> output       :    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.tsv
[Nov 03, 12:42:34] #> topk 3, thr -1.0
[Nov 03, 12:48:17] #> The # of positives: Min 3, Max 3, Mean 3.00, Median 3.0
[Nov 03, 12:48:17] #> "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.tsv" contains only pseudo-positives.
output (qrels): experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.tsv
output (triples): experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk3.jsonl
# 
# Pseudo-positive V3
topk=3
sh scripts/pseudo_positive_pretraining/msmarco_psg.label_pseudo_positives.rf.with_lp.sh ${topk} -1
[Nov 03, 13:47:35] #> Loading qrels from data/qrels.train.tsv ...
[Nov 03, 13:47:37] #> Loaded qrels for 502939 unique queries with 1.06 positives per query on average.
[Nov 03, 13:47:38] #> ranking_jsonl:    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/ranking.jsonl
[Nov 03, 13:47:38] #> output       :    experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.with_lp.tsv
[Nov 03, 13:47:38] #> topk 3, thr -1.0
[Nov 03, 13:53:11] #> The # of positives: Min 3, Max 3, Mean 3.00, Median 3.0
[Nov 03, 13:53:11] #> Add labeled positives: data/qrels.train.tsv
[Nov 03, 13:53:11] #> "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.with_lp.tsv" contains both pseudo-positives and labeled positives.
#> The # of positives: Min 4, Max 10, Mean 4.06, Median 4.0
#> The # of negatives: Min 90, Max 100, Mean 96.08, Median 96.0
output (qrels): experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.with_lp.tsv
output (triples): experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk3.with_lp.jsonl

# Pre-training using pseudo-positives: w/o KD (hard label, treating the pseudo-positives as true positives)
#TODO: gpu23-hn-pp
devices=2,3 # e.g., "0,1"
master_port=29600 # e.g., "29500"
pseudo_positive_triple=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk3.with_lp.jsonl # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv"
exp_root=experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp # e.g., "experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.kd.rf.beta0.5"
sh scripts/pseudo_positive_pretraining/msmarco_psg.pretraining.sh ${devices} ${master_port} ${pseudo_positive_triple} ${exp_root}
# validation
# mkdir -p experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
# mkdir -p experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 25000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 50000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 75000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 100000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 150000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 200000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 250000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 300000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 350000
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 400000
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 25000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/25000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 50000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/50000.log
# sh scripts/validation/msmarco_psg.sh 2 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 75000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/75000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 100000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/100000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 150000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/150000.log
# sh scripts/validation/msmarco_psg.sh 2 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 200000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/200000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 250000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/250000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 300000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/300000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 350000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/350000.log
sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp 400000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp/MSMARCO-psg/test.py/400000.log

  
# Pre-training using pseudo-positives: w/ KD (teacher is ColBERT-PRF)
#TODO: gpu45-hn-pp-prf
devices=4,5 # e.g., "0,1"
master_port=29700 # e.g., "29500"
pseudo_positive_triple=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk3.with_lp.jsonl # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv"
exp_root=experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 # e.g., "experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.kd.rf.beta0.5"
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"
sh scripts/pseudo_positive_pretraining/msmarco_psg.pretraining.kd.sh ${devices} ${master_port} ${pseudo_positive_triple} ${exp_root} ${kd_expansion_pt}
# validation
mkdir -p experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 100000
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 25000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py/25000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 50000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py/50000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 100000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py/100000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 150000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py/150000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 200000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py/200000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 250000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py/250000.log
# sh scripts/validation/msmarco_psg.sh 0 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5 300000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.prf.beta0.5/MSMARCO-psg/test.py/300000.log


# Pre-training using pseudo-positives: w/ KD (teacher is ColBERT-RF)
#TODO: gpu67-hn-pp-rf
devices=6,7 # e.g., "0,1"
master_port=29800 # e.g., "29500"
pseudo_positive_triple=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/triples.hn1.pp-topk3.with_lp.jsonl # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/pseudo_qrels.topk3.thr40.tsv"
exp_root=experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 # e.g., "experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.kd.rf.beta0.5"
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"
sh scripts/pseudo_positive_pretraining/msmarco_psg.pretraining.kd.sh ${devices} ${master_port} ${pseudo_positive_triple} ${exp_root} ${kd_expansion_pt}
# validation
mkdir -p experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py
# ./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 100000
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 25000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py/25000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 50000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py/50000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 100000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py/100000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 150000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py/150000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 200000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py/200000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 250000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py/250000.log
# sh scripts/validation/msmarco_psg.sh 1 experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5 300000 > experiments/pseudo_positive_pretraining/pretrained.b36.lr3e6.hn.pp.kd.rf.beta0.5/MSMARCO-psg/test.py/300000.log

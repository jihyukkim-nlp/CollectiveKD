# https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md
# https://github.com/castorini/docTTTTTquery



# docT5query retrieve feedback documents
# This was done in dilab003
cd /hdd/jihyuk/DataCenter/MSMARCO/docT5query
sh /workspace/GitHubCodes/anserini/target/appassembler/bin/SearchMsmarco -index lucene-index-msmarco-passage-expanded -queries /hdd/jihyuk/DataCenter/MSMARCO/queries.train.reduced.tsv  -output run.msmarco-passage-expanded.train.reduced.txt -hits 10 -threads 8
# copy to dilab4
scp jihyuk@dilab003.yonsei.ac.kr:/hdd/jihyuk/DataCenter/MSMARCO/docT5query/run.msmarco-passage-expanded.train.reduced.txt experiments/ensemble/fb_docs/docT5query.queries.train.reduced.tsv
# convert tsv file into json file, for compatibility
python -m scripts.label.pilot_test.fb_bm25.tsv_to_jsonl --tsv experiments/ensemble/fb_docs/docT5query.queries.train.reduced.tsv --output experiments/ensemble/fb_docs/docT5query.queries.train.reduced.jsonl



# Get expansion embeddings using feedback documents from docT5query
# v1: 
device=7
fb_k=10
beta=0.5
fb_clusters=24
fb_docs=3
expansion_exp=MSMARCO-psg-train-prf.fb_docT5query.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters}
fb_ranking=experiments/ensemble-loss/fb_docs/docT5query.queries.train.reduced.jsonl
sh scripts/ensemble-loss/teacher/msmarco_psg.qe.prf.sh ${device} ${fb_k} ${beta} ${fb_clusters} ${fb_docs} ${expansion_exp} ${fb_ranking}
# result: experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt
# 
# v2: 
device=7
fb_k=10
beta=1.0
fb_clusters=24
fb_docs=3
expansion_exp=MSMARCO-psg-train-prf.fb_docT5query.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters}
fb_ranking=experiments/ensemble-loss/fb_docs/docT5query.queries.train.reduced.jsonl
sh scripts/ensemble-loss/teacher/msmarco_psg.qe.prf.sh ${device} ${fb_k} ${beta} ${fb_clusters} ${fb_docs} ${expansion_exp} ${fb_ranking}
# result: experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta1.0.clusters24/label.py/2021-11-07_23.05.35/expansion.pt







# Training using ensemble of teachers
#* ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5
devices=4,5 # e.g., "0,1"
master_port=29700 # e.g., "29500"
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt
kd_expansion_pt2=experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt
sh scripts/ensemble-loss/msmarco_psg.training.kd.ensemble-loss.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt1} ${kd_expansion_pt2}
# validation
mkdir -p experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5 100000
sh scripts/validation/msmarco_psg.sh 7 experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5 100000 > experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/100000.log
# colbert-100000.dnn: MRR@10 **0.37108314003729453** (experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-08_20.10.42/ranking.tsv)
# indexing #TODO:
indexing_devices=0,1
faiss_devices=0,1
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5
step=100000
#TODO:sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}
# retrieval & reranking & evaluation
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5
step=100000
device=0
#TODO:sh scripts/ranking/msmarco_psg.ranking.sh ${exp_root} ${step} ${device}
#TODO:sh scripts/ranking/trec2019_psg.ranking.sh ${exp_root} ${step} ${device}
#TODO:sh scripts/ranking/trec2020_psg.ranking.sh ${exp_root} ${step} ${device}



#* ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5
devices=0,1 # e.g., "0,1"
master_port=29500 # e.g., "29500"
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt
sh scripts/ensemble-loss/msmarco_psg.training.kd.ensemble-loss.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt1}
# validation #TODO:
mkdir -p experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5 150000
sh scripts/validation/msmarco_psg.sh 7 experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5 150000 > experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5/MSMARCO-psg/test.py/150000.log
# colbert-150000.dnn: MRR@10 **0.3701702142174925** (experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-08_23.00.26/ranking.tsv)
# indexing #TODO:
indexing_devices=0,1,2,3,4,5,6,7
faiss_devices=0,1
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5
step=150000
#TODO:sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}
# retrieve & rerank & evaluation #TODO:
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta0.5
step=150000
device=0
#TODO:sh scripts/ranking/msmarco_psg.ranking.sh ${exp_root} ${step} ${device}
#TODO:sh scripts/ranking/trec2019_psg.ranking.sh ${exp_root} ${step} ${device}
#TODO:sh scripts/ranking/trec2020_psg.ranking.sh ${exp_root} ${step} ${device}



#* ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0
devices=2,3 # e.g., "0,1"
master_port=29600 # e.g., "29500"
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta1.0.clusters24/label.py/2021-11-07_23.05.35/expansion.pt
sh scripts/ensemble-loss/msmarco_psg.training.kd.ensemble-loss.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt1}
# validation #TODO:
mkdir -p experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_dilab4.sh experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0 150000
sh scripts/validation/msmarco_psg.sh 4 experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0 150000 > experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0/MSMARCO-psg/test.py/150000.log
# colbert-150000.dnn: MRR@10 **0.36637518761086124** (experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0/MSMARCO-psg/test.py/2021-11-09_06.10.33/ranking.tsv)
# indexing #TODO:
indexing_devices=0,1,2,3,4,5,6,7
faiss_devices=0,1
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0
step=150000
#TODO:sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}
# retrieve & rerank & evaluation #TODO:
exp_root=experiments/ensemble-loss/finetuned.b36.lr3e6.hn.kd.t1.prf.docT5query.beta1.0
step=150000
device=0
#TODO:sh scripts/ranking/msmarco_psg.ranking.sh ${exp_root} ${step} ${device}
#TODO:sh scripts/ranking/trec2019_psg.ranking.sh ${exp_root} ${step} ${device}
#TODO:sh scripts/ranking/trec2020_psg.ranking.sh ${exp_root} ${step} ${device}




#?@ debugging
devices=1 # e.g., "0,1"
master_port=29500 # e.g., "29500"
exp_root=experiments/ensemble-loss/debugging
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt
# kd_expansion_pt2=""
kd_expansion_pt2=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt
# kd_expansion_pt3=""
kd_expansion_pt3=experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt
sh scripts/ensemble-loss/msmarco_psg.training.kd.ensemble-loss.debugging.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt1} ${kd_expansion_pt2} ${kd_expansion_pt3}
# sh scripts/ensemble-loss/msmarco_psg.training.kd.ensemble-loss.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt1} ${kd_expansion_pt2} ${kd_expansion_pt3}

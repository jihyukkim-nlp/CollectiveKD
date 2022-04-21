"""
Gather collective feedback for queries using pre-trained ColBERT (trained using labeled positives and BM25 negatives without knowledge distillation)
1. Indexing using pre-trained ColBERT
2. NN Search for queries
3. Obtain collective feedback as representative and discriminative feedback term embeddings.
"""


# 1. Indexing
# This takes approximately an hour.
indexing_devices=0,1,2,3,4,5,6,7
faiss_devices=0,1
exp_root=experiments/colbert.teacher
checkpoint=data/checkpoints/colbert.teacher.dnn
sh scripts/gather_collective_feedback/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${checkpoint}
# cat experiments/colbert.teacher/MSMARCO-psg/index.py/2022-01-03_03.11.11/logs/elapsed.txt > 1625.8355560302734 seconds (27.097259267171223 minutes)
# cat experiments/colbert.teacher/MSMARCO-psg/index_faiss.py/2022-01-03_03.38.25/logs/elapsed.txt > 829.5255172252655 seconds (13.825425287087758 minutes)
# du -hs experiments/colbert.teacher/MSMARCO-psg/index.py/MSMARCO.L2.32x200k # 163G



# 2. NN Search
# This takes 21 hours; 5 hours for ANN search and 16 hours for Exact-NN search
device=0
exp_root=experiments/colbert.teacher
checkpoint=data/checkpoints/colbert.teacher.dnn
sh scripts/gather_collective_feedback/nn_search.sh ${device} ${exp_root} ${checkpoint}
"""
ls experiments/colbert.teacher/MSMARCO-psg-HN
label.py  retrieve.py

# ANN Search
cat experiments/colbert.teacher/MSMARCO-psg-HN/retrieve.py/2022-01-03_03.54.50/logs/elapsed.txt
18793.31041622162 # approximately 5 hourss

du -hs experiments/colbert.teacher/MSMARCO-psg-HN/retrieve.py/2022-01-03_03.54.50/*
87G     experiments/colbert.teacher/MSMARCO-psg-HN/retrieve.py/2022-01-03_03.54.50/queries.train.chunks50000
87G     experiments/colbert.teacher/MSMARCO-psg-HN/retrieve.py/2022-01-03_03.54.50/unordered.tsv

du -hs experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking*
4.3G    experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl
5.6G    experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.tsv

wc -l experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl
808731 experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl

wc -l data/msmarco-pass/queries.train.tsv
808731 data/msmarco-pass/queries.train.tsv

cat experiments/colbert.teacher/MSMARCO-psg-HN/label.py/*/logs/elapsed.txt
3850.3420116901398
3953.6259067058563
3787.3294129371643
3586.220416545868
3773.4860310554504
3588.2408940792084
3386.534867286682
3511.2558228969574
3424.6887283325195
3460.0446739196777
3518.850188732147
3676.1061050891876
3634.64408326149
3639.3854897022247
3870.6027903556824
3821.16210603714
793.2533140182495

tot_t=0
for t in $(cat experiments/colbert.teacher/MSMARCO-psg-HN/label.py/*/logs/elapsed.txt);do
    tot_t=$(perl -e "print ${tot_t} + ${t}")
done
echo ${tot_t}
59275.7728426457 # 16.465492456290472 hourss

results: experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl
"""


# 3. Obtain collective feedback 
# This takes approximately 8-9 hours.
# (docs 3, clusters 24, k 10, beta 1.0)
device=0
exp_root=experiments/colbert.teacher
checkpoint=data/checkpoints/colbert.teacher.dnn
fb_ranking=experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl
fb_k=10
beta=1.0
fb_clusters=24
fb_docs=3
sh scripts/gather_collective_feedback/gather_collective_feedback.sh ${device} ${exp_root} ${checkpoint} ${fb_ranking} ${fb_docs} ${fb_clusters} ${fb_k} ${beta}
"""
cat experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta1.0/label.py/2022-01-04_21.17.26/logs/elapsed.txt 
30826.6286611557 # => 8.562952405876583 hours

du -hs experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta1.0/label.py/2022-01-04_21.17.26/expansion.pt
4.2G    experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta1.0/label.py/2022-01-04_21.17.26/expansion.pt

results: experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta1.0/label.py/2022-01-04_21.17.26/expansion.pt
"""
# 
#TODO: gpu0
# (docs 3, clusters 24, k 10, beta 0.5)
device=0
exp_root=experiments/colbert.teacher
checkpoint=data/checkpoints/colbert.teacher.dnn
fb_ranking=experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl
fb_k=10
beta=0.5
fb_clusters=24
fb_docs=3
sh scripts/gather_collective_feedback/gather_collective_feedback.sh ${device} ${exp_root} ${checkpoint} ${fb_ranking} ${fb_docs} ${fb_clusters} ${fb_k} ${beta}
# results: #TODO



# Remove index
#TODO: waiting for gathering collective feedback complete
rm -r -v experiments/colbert.teacher/MSMARCO-psg/index.py/MSMARCO.L2.32x200k/
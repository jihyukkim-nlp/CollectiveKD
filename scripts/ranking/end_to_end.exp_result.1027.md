# Baseline: ColBERT (copied from paper https://arxiv.org/pdf/2004.12832.pdf)
Re-ranking
   MRR@10: 0.349
End-to-end ranking
   MRR@10 = 0.360
   Recall@50 = 0.829
   Recall@200 = 0.923
   Recall@1000 = 0.968

# Baseline: ColBERT (our implementation - trained using in-batch negatives) (colbert.teacher.dnn)
Re-ranking
   MRR@10: 0.354
End-to-end ranking
   MRR@10 = 0.367
   Recall@50 = 0.833
   Recall@200 = 0.925
   Recall@1000 = 0.967

# Baseline: ColBERT-PRF (ReRanker, beta=0.5) (copied from paper: https://arxiv.org/pdf/2106.11251.pdf)
End-to-end ranking (on TREC 2019)
   MAP = 0.5026
   NDCG@10 = 0.7409
   Recall@1000 = 0.7977
   MRR@10 = 0.8897

End-to-end ranking (on TREC 2020)
   MAP = 0.5063
   NDCG@10 = 0.7161
   Recall@1000 = 0.8443
   MRR@10 = 0.8439



# Newly Experimented (Fine-tuning ColBERT)

## Baseline. finetuned.b36.lr3e6.hn (HN fine-tuning)
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 

Validation performance (on re-reanking task)
**colbert-50000.dnn** : MRR@10 **0.36058653977350225** (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-09_14.09.52)

End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/rerank.py/2021-10-12_11.26.35/ranking.tsv)
[Oct 12, 11:46:55] #> MRR@10 = 0.37442949242734364
[Oct 12, 11:46:55] #> MRR@100 = 0.38450792961770175
[Oct 12, 11:46:55] #> Recall@50 = 0.8424546322827126
[Oct 12, 11:46:55] #> Recall@200 = 0.9235673352435532
[Oct 12, 11:46:55] #> Recall@1000 = 0.9691260744985676

TREC 2019: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn/TREC2019-psg/rerank.py/2021-10-12_14.24.55/ranking.tsv)
{'AP(rel=2)@1000': 0.48508580264653356, 'nDCG@10': 0.7414371749888692, 'RR(rel=2)@10': 0.9031007751937985, 'R(rel=2)@100': 0.6262149523371159, 'R(rel=2)@1000': 0.8378453338575937, 'nDCG@50': 0.6651141257207117, 'nDCG@100': 0.6546823830897742, 'nDCG@200': 0.6682710830890339, 'nDCG@500': 0.7006364918196664, 'nDCG@1000': 0.717662830416884, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/rerank.py/2021-10-12_14.24.58/ranking.tsv)
{'AP(rel=2)@1000': 0.49370822262673547, 'nDCG@10': 0.6998388784820837, 'RR(rel=2)@10': 0.8370370370370371, 'R(rel=2)@100': 0.7405625934528436, 'R(rel=2)@1000': 0.8670388401182964, 'nDCG@50': 0.6520576880566012, 'nDCG@100': 0.6495910621986439, 'nDCG@200': 0.6762072849837231, 'nDCG@500': 0.7025032136354434, 'nDCG@1000': 0.7159782428303274, 'NumRet': 54000.0, 'num_q': 54.0}


## Baseline. finetuned.b36.lr3e6.hn.up (self-training)
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - UP (Unlabeled Positives) from pre-trained ColBERT

Validation performance (on re-reanking task)
**colbert-400000.dnn**: MRR@10 **0.360915597853278** (experiments/finetuned.b36.lr3e6.hn.up/MSMARCO-psg/test.py/2021-10-09_06.32.30)

## Baseline. finetuned.b36.lr3e6.hn.kd (TODO: born again network)
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT

Validation performance (on re-reanking task)

## Ours. finetuned.b36.lr3e6.hn.up_qe
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - UP (Unlabeled Positives) from pre-trained ColBERT using expanded query

Validation performance (on re-reanking task)
**colbert-300000.dnn**: MRR@10 **0.3632926274616826** (experiments/finetuned.b36.lr3e6.hn.up_qe/MSMARCO-psg/test.py/2021-10-08_23.43.22)

## Ours. finetuned.b36.lr3e6.hn.kd_qe
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query

Validation performance (on re-reanking task)
**colbert-300000.dnn**: MRR@10 **0.3645621844726432** (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-09_15.04.47)

End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/rerank.py/2021-10-12_14.04.01/ranking.tsv)
[Oct 12, 14:11:01] #> MRR@10 = 0.3802522513303315
[Oct 12, 14:11:01] #> MRR@100 = 0.390437271878556
[Oct 12, 14:11:01] #> Recall@50 = 0.8478510028653296
[Oct 12, 14:11:01] #> Recall@200 = 0.9349450811843363
[Oct 12, 14:11:01] #> Recall@1000 = 0.9717884431709649

TREC 2019: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe/TREC2019-psg/rerank.py/2021-10-12_14.38.06/ranking.tsv )
{'ndcg_cut_10': 0.73821514263214, 'ndcg_cut_200': 0.6775029129854387, 'map_cut_1000': 0.4919384904436252, 'recall_100': 0.5384174306280657, 'recall_200': 0.6420607462757519, 'recall_500': 0.7400378534089058, 'recall_1000': 0.79055624856861}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe/TREC2020-psg/rerank.py/2021-10-12_14.38.29/ranking.tsv)
{'ndcg_cut_10': 0.7000146727906563, 'ndcg_cut_200': 0.6753943148077853, 'map_cut_1000': 0.49742549437038075, 'recall_100': 0.6033897423400646, 'recall_200': 0.6788182266910773, 'recall_500': 0.7389882775546902, 'recall_1000': 0.7840222968869982}

## Ours. finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans

Validation performance (on re-reanking task)
colbert-100000.dnn: MRR@10 **0.3704880724973852** (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-18_05.21.45)

End-to-end ranking performance (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/rerank.py/2021-10-18_20.26.26/ranking.tsv)
[Oct 18, 20:51:33] #> MRR@10 = 0.38362577886933164
[Oct 18, 20:51:33] #> MRR@100 = 0.3938578814457015
[Oct 18, 20:51:33] #> Recall@50 = 0.8514446036294173
[Oct 18, 20:51:33] #> Recall@200 = 0.9334527220630372
[Oct 18, 20:51:33] #> Recall@1000 = 0.9712392550143267

TREC 2019: End-to-end ranking performance (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/TREC2019-psg/rerank.py/2021-10-18_20.51.32/ranking.tsv)
{'AP(rel=2)@1000': 0.49015249398084093, 'nDCG@10': 0.7529632436133189, 'RR(rel=2)@10': 0.9147286821705427, 'R(rel=2)@100': 0.6320037547135006, 'R(rel=2)@1000': 0.8371122732659264, 
'nDCG@50': 0.6815404049055414, 'nDCG@100': 0.6620803204597868, 'nDCG@200': 0.6759445722105133, 'nDCG@500': 0.7091573691445375, 'nDCG@1000': 0.7265033537962282, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/TREC2020-psg/rerank.py/2021-10-18_20.54.23/ranking.tsv)
{'AP(rel=2)@1000': 0.49690454563012737, 'nDCG@10': 0.708967344391837, 'RR(rel=2)@10': 0.8302469135802468, 'R(rel=2)@100': 0.7480946306254345, 'R(rel=2)@1000': 0.8726743090322439,
'nDCG@50': 0.6621717538904349, 'nDCG@100': 0.6592712569360509, 'nDCG@200': 0.6834181598602931, 'nDCG@500': 0.7112882404292719, 'nDCG@1000': 0.7254737124703, 'NumRet': 54000.0, 'num_q': 54.0}



## Ours. finetuned.b36.lr3e6.hn.up_qe.kd_qe
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - UP (Unlabeled Positives) from pre-trained ColBERT using expanded query
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query

Validation performance (on re-reanking task)
**colbert-400000.dnn**: MRR@10 **0.3621587187883749** (experiments/finetuned.b36.lr3e6.hn.up_qe.kd_qe/MSMARCO-psg/test.py/2021-10-09_14.21.51)





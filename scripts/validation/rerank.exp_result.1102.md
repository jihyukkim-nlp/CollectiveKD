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

# Pivot samples (MRR@10)
2500 -> 36.9
3500 -> 36.4
4000/4100 -> 36.0
4400 -> 35.9
4500 -> 35.8
4700 -> 35.8
4800 -> 35.9
4900 -> 36.0
5000 -> 36.0
5200 -> 36.1
5500 -> 36.08
5800 -> 36.2
6000 -> 36.3
6979 -> 36.4

# Newly Experimented (Fine-tuning ColBERT; bsize 18, # of HN 4)

## Baseline. finetuned.b18.lr3e6.hn.n4 (HN fine-tuning)
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.3644126074498563 (experiments/finetuned.b18.lr3e6.hn.n4/MSMARCO-psg/test.py/2021-10-19_01.44.31)
**colbert-50000.dnn** : MRR@10 **0.36822144903806786** (experiments/finetuned.b18.lr3e6.hn.n4/MSMARCO-psg/test.py/2021-10-19_01.45.00)
colbert-75000.dnn : MRR@10 0.3641269272752084 (experiments/finetuned.b18.lr3e6.hn.n4/MSMARCO-psg/test.py/2021-10-19_01.45.14)
colbert-100000.dnn: MRR@10 0.36816726929549315 (experiments/finetuned.b18.lr3e6.hn.n4/MSMARCO-psg/test.py/2021-10-19_04.59.23)
colbert-150000.dnn: MRR@10 0.36390321553645305 (experiments/finetuned.b18.lr3e6.hn.n4/MSMARCO-psg/test.py/2021-10-19_05.01.19)
colbert-200000.dnn: MRR@10 0.3613337995178966 (experiments/finetuned.b18.lr3e6.hn.n4/MSMARCO-psg/test.py/2021-10-19_10.13.43)

## Ours. finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - KD temperature: 0.25

Validation performance (on re-reanking task)
colbert-10000.dnn : MRR@10 0.36213768363123655 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-17_18.15.45)
colbert-25000.dnn : MRR@10 0.36440260153727216 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-17_20.21.54)
colbert-50000.dnn : MRR@10 0.3677560603993272 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-17_22.56.54)
colbert-75000.dnn : MRR@10 0.36699339382362345 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-18_05.17.44)
colbert-100000.dnn: MRR@10 **0.3704880724973852** (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-18_05.21.45)
colbert-150000.dnn: MRR@10 0.3680415700186469 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-19_08.13.10)
colbert-200000.dnn: MRR@10 0.3646333628962565 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-19_15.31.23)
colbert-250000.dnn: MRR@10 0.36235627871014664 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-20_07.34.45)
colbert-300000.dnn: MRR@10 0.35822895347250666 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.n4/MSMARCO-psg/test.py/2021-10-20_07.35.11)


# Newly Experimented (Fine-tuning ColBERT)

## Baseline. finetuned.b36.lr3e6.hn (HN fine-tuning)
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.35858188929822105 (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-10_08.52.32)
**colbert-50000.dnn** : MRR@10 **0.36058653977350225** (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-09_14.09.52)
colbert-75000.dnn : MRR@10 0.35628143903215514 (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-10_08.53.15)
colbert-100000.dnn: MRR@10 0.3583941647337059 (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-09_17.25.03)
colbert-150000.dnn: MRR@10 0.3565283462955376 (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-10_13.32.11)
colbert-200000.dnn: MRR@10 0.3486692136262331 (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-10_05.31.13)
colbert-300000.dnn: MRR@10 0.3491918520034562 (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-10_14.41.19)
colbert-400000.dnn: MRR@10 0.3457999613407924 (experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/test.py/2021-10-11_03.06.48)

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) instead of RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt)

colbert-25000.dnn : MRR@10 0.361314015099831 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-25_10.12.00)
colbert-50000.dnn : MRR@10 0.3660505071178425 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-25_13.07.07)
colbert-75000.dnn : MRR@10 0.36457048483194626 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-25_16.51.35)
colbert-100000.dnn: MRR@10 0.36907064401691875 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-25_19.29.33)
colbert-**150000**.dnn: MRR@10 **0.3704569177241098** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-26_04.18.23)
colbert-200000.dnn: MRR@10 0.37015367035066177 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-26_07.40.39)
colbert-250000.dnn: MRR@10 0.3673625892572882 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-26_14.35.41)
colbert-300000.dnn: MRR@10 0.3678467389821255 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-26_21.01.48)

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) instead of RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt)
- The 2nd iteration KD training using better student and teacher
   - train triples (``triples``): hard negatives are obtained by previously fine-tuned ColBERT
   - teacher (``kd_expansion_pt``): expansion embeddings are obtained by previously fine-tuned ColBERT

colbert-25000.dnn : MRR@10 0.36792434165643323 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/2021-10-31_17.11.09)
colbert-50000.dnn : MRR@10 0.36648081821076084 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/2021-10-31_20.53.52)
colbert-75000.dnn : MRR@10 0.3649557124664578 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/2021-10-31_23.35.05)
colbert-100000.dnn: MRR@10 0.36687679083094576 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/2021-11-01_02.28.53)
colbert-**150000**.dnn: MRR@10 **0.36830200800472995** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5.2nd_kd/MSMARCO-psg/test.py/2021-11-01_10.06.49)


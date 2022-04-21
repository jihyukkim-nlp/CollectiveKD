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

## Baseline. finetuned.b36.lr3e6.hn.up (self-training)
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - UP (Unlabeled Positives) from pre-trained ColBERT

Validation performance (on re-reanking task)
colbert-50000.dnn : MRR@10 0.3539616591622325 (experiments/finetuned.b36.lr3e6.hn.up/MSMARCO-psg/test.py/2021-10-08_12.26.58) 
colbert-100000.dnn: MRR@10 0.35856392413698934 (experiments/finetuned.b36.lr3e6.hn.up/MSMARCO-psg/test.py/2021-10-07_23.34.26)
colbert-200000.dnn: MRR@10 0.35665660390230586 (experiments/finetuned.b36.lr3e6.hn.up/MSMARCO-psg/test.py/2021-10-08_08.39.04)
colbert-300000.dnn: MRR@10 0.3588918451812441 (experiments/finetuned.b36.lr3e6.hn.up/MSMARCO-psg/test.py/2021-10-08_23.44.00)
**colbert-400000.dnn**: MRR@10 **0.360915597853278** (experiments/finetuned.b36.lr3e6.hn.up/MSMARCO-psg/test.py/2021-10-09_06.32.30)

## Baseline. finetuned.b36.lr3e6.hn.kd (born again network)
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.35588137308409507 (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-14_13.26.02)
colbert-50000.dnn : MRR@10 0.3556772183563007 (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-14_13.21.30)
colbert-75000.dnn : MRR@10 0.3567736617091912 (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-14_15.57.19)
colbert-100000.dnn: MRR@10 0.3568569495611038 (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-14_18.56.12)
colbert-150000.dnn: MRR@10 0.35646364897439325 (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-15_00.48.39)
colbert-200000.dnn: MRR@10 0.35690061172511023 (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-15_06.51.38) 
colbert-300000.dnn: MRR@10 **0.35876063128212077** (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-17_06.46.53) 
colbert-400000.dnn: MRR@10 0.35875369536544294 (experiments/finetuned.b36.lr3e6.hn.kd/MSMARCO-psg/test.py/2021-10-17_06.47.19) 

## Ours. finetuned.b36.lr3e6.hn.up_qe
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - UP (Unlabeled Positives) from pre-trained ColBERT using expanded query

Validation performance (on re-reanking task)
colbert-50000.dnn : MRR@10 0.357447241551826 (experiments/finetuned.b36.lr3e6.hn.up_qe/MSMARCO-psg/test.py/2021-10-08_12.24.55)
colbert-100000.dnn: MRR@10 0.35996804930185966 (experiments/finetuned.b36.lr3e6.hn.up_qe/MSMARCO-psg/test.py/2021-10-07_23.34.51)
colbert-200000.dnn: MRR@10 0.359871287579024 (experiments/finetuned.b36.lr3e6.hn.up_qe/MSMARCO-psg/test.py/2021-10-08_08.39.42)
**colbert-300000.dnn**: MRR@10 **0.3632926274616826** (experiments/finetuned.b36.lr3e6.hn.up_qe/MSMARCO-psg/test.py/2021-10-08_23.43.22)
colbert-400000.dnn: MRR@10 0.3603348569609314 (experiments/finetuned.b36.lr3e6.hn.up_qe/MSMARCO-psg/test.py/2021-10-09_06.33.05)

## Ours. finetuned.b36.lr3e6.hn.kd_qe
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query

Validation performance (on re-reanking task)
colbert-50000.dnn : MRR@10 0.36039546095420094 (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-08_08.40.17)
colbert-100000.dnn: MRR@10 0.36374039204984787 (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-09_02.59.10)
colbert-150000.dnn: MRR@10 0.3612171396734439 (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-10_22.01.45)
colbert-200000.dnn: MRR@10 0.36168821348978936 (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-09_02.59.40)
**colbert-300000.dnn**: MRR@10 **0.3645621844726432** (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-09_15.04.47)
colbert-400000.dnn: MRR@10 0.3637189589302766 (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-10_03.24.28)
colbert-450000.dnn: MRR@10 0.36258556192295455 (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-10_13.33.32) 
colbert-500000.dnn: MRR@10 0.36277305908036533 (experiments/finetuned.b36.lr3e6.hn.kd_qe/MSMARCO-psg/test.py/2021-10-10_21.48.36) 

## Ours. finetuned.b36.lr3e6.kd_qe_kmeans
- Training using new train triples: 
   - BM25 negatives
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans

Validation performance (on re-reanking task)
colbert-100000.dnn: MRR@10 0.3450798767453495 (experiments/finetuned.b36.lr3e6.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-22_08.19.36)
colbert-150000.dnn: MRR@10 0.3486766043571211 (experiments/finetuned.b36.lr3e6.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-22_08.28.18)
colbert-200000.dnn: MRR@10 () ()
colbert-250000.dnn: MRR@10 () ()
colbert-300000.dnn: MRR@10 () ()
colbert-350000.dnn: MRR@10 () ()
colbert-400000.dnn: MRR@10 () ()

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.36033281029699277 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-17_06.32.11)
**colbert-50000.dnn** : MRR@10 **0.36324219993632545** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-17_11.24.50)
colbert-75000.dnn : MRR@10 0.3604472529221805 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-17_11.54.50)
colbert-100000.dnn: MRR@10 0.36250756128621403 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-17_14.43.06)
colbert-150000.dnn: MRR@10 0.36011535225360425 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-17_21.12.41)
colbert-200000.dnn: MRR@10 0.35463489789421027 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-18_05.07.52)

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

## Ours. finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - KD temperature: 0.50

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.3628852844862876 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-20_02.06.25)
**colbert-50000.dnn** : MRR@10 **0.3694713921862921** (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-20_02.06.56)
colbert-75000.dnn : MRR@10 0.36582622686132726 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-20_02.08.18)
colbert-100000.dnn: MRR@10 0.36859200891435806 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-20_07.25.12)
colbert-150000.dnn: MRR@10 0.36756759676172257 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-20_12.58.45)
colbert-200000.dnn: MRR@10 0.36317755946695707 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-21_04.20.43)
colbert-250000.dnn: MRR@10 0.36117205621503573 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-21_04.23.28)
colbert-300000.dnn: MRR@10 0.3589795106199118 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t50.n4/MSMARCO-psg/test.py/2021-10-21_07.45.46)

## Ours. finetuned.b18.lr3e6.hn.kd_qe_kmeans.t100.n4
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - KD temperature: 1.0

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.3634743257379354 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t100.n4/MSMARCO-psg/test.py/2021-10-21_00.27.43)
**colbert-50000.dnn** : MRR@10 **0.37075152362759856** (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t100.n4/MSMARCO-psg/test.py/2021-10-21_00.28.36)
colbert-75000.dnn : MRR@10 0.36822105107563574 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t100.n4/MSMARCO-psg/test.py/2021-10-21_00.28.54)
colbert-100000.dnn: MRR@10 0.36992535361804746 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t100.n4/MSMARCO-psg/test.py/2021-10-21_04.25.16)
colbert-150000.dnn: MRR@10 0.368895313139582 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t100.n4/MSMARCO-psg/test.py/2021-10-21_07.42.59)
colbert-200000.dnn: MRR@10 0.3649794196570701 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t100.n4/MSMARCO-psg/test.py/2021-10-21_11.05.02)

## Ours. finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - KD temperature: 2.0

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.3633720493928225 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4/MSMARCO-psg/test.py/2021-10-21_12.29.05)
colbert-50000.dnn : MRR@10 0.3693636580706783 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4/MSMARCO-psg/test.py/2021-10-21_14.26.27)
colbert-75000.dnn : MRR@10 0.36772291581388994 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4/MSMARCO-psg/test.py/2021-10-22_02.22.29)
**colbert-100000.dnn**: MRR@10 **0.3707967776413334** (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4/MSMARCO-psg/test.py/2021-10-22_02.22.16)
colbert-150000.dnn: MRR@10 0.3685236730795462 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4/MSMARCO-psg/test.py/2021-10-22_05.00.48)
colbert-200000.dnn: MRR@10 (gpu0) (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4/MSMARCO-psg/test.py/2021-10-22_10.43.54)

## Ours. finetuned.b36.lr3e6.hn.up_qe.kd_qe
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - UP (Unlabeled Positives) from pre-trained ColBERT using expanded query
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query

Validation performance (on re-reanking task)
colbert-50000.dnn : MRR@10 0.3565646745804336 (experiments/finetuned.b36.lr3e6.hn.up_qe.kd_qe/MSMARCO-psg/test.py/2021-10-09_02.58.13)
colbert-100000.dnn: MRR@10 0.358017010051394 (experiments/finetuned.b36.lr3e6.hn.up_qe.kd_qe/MSMARCO-psg/test.py/2021-10-07_23.34.59)
colbert-200000.dnn: MRR@10 0.35915262427798234 (experiments/finetuned.b36.lr3e6.hn.up_qe.kd_qe/MSMARCO-psg/test.py/2021-10-08_12.04.33)
colbert-300000.dnn: MRR@10 0.36050291081093355 (experiments/finetuned.b36.lr3e6.hn.up_qe.kd_qe/MSMARCO-psg/test.py/2021-10-08_23.42.34)
**colbert-400000.dnn**: MRR@10 **0.3621587187883749** (experiments/finetuned.b36.lr3e6.hn.up_qe.kd_qe/MSMARCO-psg/test.py/2021-10-09_14.21.51)
colbert-572000.dnn: MRR@10 0.35936820621276255 (experiments/finetuned.b36.lr3e6.hn.up_qe.kd_qe/MSMARCO-psg/test.py/2021-10-10_08.22.38)





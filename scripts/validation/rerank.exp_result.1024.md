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

## Ours. finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty3.n4
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - KD temperature: 0.25
   - Giving score penalty for labeled passages (--kd_penalty 3)

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.3626179105835268 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty3.n4/MSMARCO-psg/test.py/2021-10-23_02.19.02)
colbert-50000.dnn : MRR@10 0.3668056692591076 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty3.n4/MSMARCO-psg/test.py/2021-10-23_02.14.54)
colbert-75000.dnn : MRR@10 0.3664471619593395 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty3.n4/MSMARCO-psg/test.py/2021-10-23_05.29.57)
colbert-100000.dnn: MRR@10 0.3695736685314048 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty3.n4/MSMARCO-psg/test.py/2021-10-23_08.47.59)
colbert-150000.dnn: MRR@10 0.3684706303724926 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty3.n4/MSMARCO-psg/test.py/2021-10-24_03.54.36)

## Ours. finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty6.n4
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - Using 4 negative samples per query (--bsize 18, instead of --bsize 36)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - KD temperature: 0.25
   - Giving score penalty for labeled passages (--kd_penalty 6)

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 () ()
colbert-50000.dnn : MRR@10 0.36799688452267326 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty6.n4/MSMARCO-psg/test.py/2021-10-23_02.15.09)
colbert-75000.dnn : MRR@10 0.3660730772729342 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty6.n4/MSMARCO-psg/test.py/2021-10-23_05.30.17)
colbert-100000.dnn: MRR@10 0.36943881611861584 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty6.n4/MSMARCO-psg/test.py/2021-10-23_08.48.22)
colbert-150000.dnn: MRR@10 0.36636199799881747 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty6.n4/MSMARCO-psg/test.py/2021-10-24_03.55.19)

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
colbert-200000.dnn: MRR@10 0.3650610019557009 (experiments/finetuned.b18.lr3e6.hn.kd_qe_kmeans.t200.n4/MSMARCO-psg/test.py/2021-10-22_10.43.54)

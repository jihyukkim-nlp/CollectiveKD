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

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.k10.beta1.0.clusters10/label.py/2021-10-17_01.17.56/expansion.pt)

colbert-25000.dnn : MRR@10 0.36033281029699277 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-24_13.02.13)
colbert-50000.dnn : MRR@10 **0.36324219993632545** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-24_13.02.26)
colbert-75000.dnn : MRR@10 0.3604472529221805 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-24_13.02.38)
colbert-100000.dnn: MRR@10 0.36250756128621403 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-24_09.30.10)
colbert-150000.dnn: MRR@10 0.36011535225360425 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-24_09.30.56)
colbert-200000.dnn: MRR@10 0.35463489789421027 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-24_09.31.22)

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt)

colbert-25000.dnn : MRR@10 0.36237242461454494 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/MSMARCO-psg/test.py/2021-10-27_12.49.48)
colbert-50000.dnn : MRR@10 0.36420669031700553 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/MSMARCO-psg/test.py/2021-10-27_15.41.20)
colbert-75000.dnn : MRR@10 0.3619485377723196 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/MSMARCO-psg/test.py/2021-10-27_18.51.30)
colbert-100000.dnn: MRR@10 0.36471551371264826 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/MSMARCO-psg/test.py/2021-10-27_21.40.24)
colbert-**150000**.dnn: MRR@10 **0.3650446854959745** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/MSMARCO-psg/test.py/2021-10-28_05.09.11)

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF instead of RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt)

colbert-25000.dnn : MRR@10 0.3613496043116384 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-24_16.30.24)
colbert-50000.dnn : MRR@10 0.36562275435484604 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-24_16.32.49)
colbert-75000.dnn : MRR@10 0.3636256651657802 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-24_19.45.31)
colbert-100000.dnn: MRR@10 0.36564287988356636 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-24_23.08.17)
colbert-**150000**.dnn: MRR@10 **0.36830644244326183** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-25_06.33.55)
colbert-200000.dnn: MRR@10 0.366268363123663 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-25_10.55.02)
colbert-250000.dnn: MRR@10 0.3667354573156865 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-25_16.51.09)
colbert-300000.dnn: MRR@10 0.3672107381634603 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-26_04.17.30)
colbert-350000.dnn: MRR@10 0.3669907786419248 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-26_05.28.42)
colbert-400000.dnn: MRR@10 0.366535623322873 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-26_12.07.58)

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

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF with RF, by appending RF docs in front of PRF docs (beta 0.5) (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_prepend_rf.docs3.k10.beta0.5.clusters24/label.py/2021-10-30_15.08.50/expansion.pt)

colbert-25000.dnn : MRR@10 0.3625139286851314 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5/MSMARCO-psg/test.py/2021-10-30_16.16.58)
colbert-50000.dnn : MRR@10 0.36445496202301336 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5/MSMARCO-psg/test.py/2021-10-30_19.10.19)
colbert-75000.dnn : MRR@10 0.36336033792695643 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5/MSMARCO-psg/test.py/2021-10-30_21.48.56)
colbert-100000.dnn: MRR@10 0.36499835129849345 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5/MSMARCO-psg/test.py/2021-10-31_01.10.31)
colbert-**150000**.dnn: MRR@10 **0.366022592895802** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5/MSMARCO-psg/test.py/2021-10-31_13.48.58)
colbert-200000.dnn: MRR@10 0.3640323259198614 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_prepend_rf-beta0.5/MSMARCO-psg/test.py/2021-10-31_13.49.03)

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_then_prf
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) (obtained by RF), along with RF (beta 0.5) (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf_then_prf.rf-k10.beta0.5.clusters10.prf-docs3.k10.beta0.5.clusters24/label.py/expansion.pt)

colbert-25000.dnn : MRR@10 0.36124903351980736 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_then_prf/MSMARCO-psg/test.py/2021-10-28_15.06.26)
colbert-50000.dnn : MRR@10 0.36402851685086646 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_then_prf/MSMARCO-psg/test.py/2021-10-28_16.29.29)
colbert-75000.dnn : MRR@10 0.3634549392823036 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_then_prf/MSMARCO-psg/test.py/2021-10-28_19.34.29)
colbert-**100000**.dnn: MRR@10 **0.3653957452130798** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_then_prf/MSMARCO-psg/test.py/2021-10-28_22.25.14)
colbert-150000.dnn: MRR@10 0.36525463910492556 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_then_prf/MSMARCO-psg/test.py/2021-10-29_04.44.02)
colbert-200000.dnn: MRR@10 0.3633196889070816 (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_then_prf/MSMARCO-psg/test.py/2021-10-29_12.50.00)

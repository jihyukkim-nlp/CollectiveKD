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

TREC 2019 
{'nDCG@10': 0.6996191676961366, 'nDCG@25': 0.6611577340895359, 'nDCG@50': 0.6489031290874794, 'nDCG@100': 0.6445046665392238, 'nDCG@200': 0.6575805267886845, 'nDCG@500': 0.6897259199696175, 'nDCG@1000': 0.7122704288228963, 'R(rel=2)@3': 0.13427849571086078, 'R(rel=2)@5': 0.1841068344211011, 'R(rel=2)@10': 0.2875825212503319, 'R(rel=2)@25': 0.41898498969474013, 'R(rel=2)@50': 0.533461722369848, 'R(rel=2)@100': 0.6350332824018848, 'R(rel=2)@200': 0.7204932770174376, 'R(rel=2)@1000': 0.8370813777761269, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.7286821705426357, 'P(rel=2)@5': 0.7069767441860467, 'P(rel=2)@10': 0.6325581395348838, 'P(rel=2)@25': 0.46790697674418597, 'P(rel=2)@50': 0.3572093023255815, 'P(rel=2)@100': 0.2590697674418605, 'P(rel=2)@200': 0.16999999999999998, 'P(rel=2)@1000': 0.043837209302325555, 'AP(rel=2)@1000': 0.4753639977588095, 'RR(rel=2)@10': 0.8330103359173127, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020 
{'nDCG@10': 0.674625300218906, 'nDCG@25': 0.6404223915640629, 'nDCG@50': 0.628640636810472, 'nDCG@100': 0.6311473087514119, 'nDCG@200': 0.6521529878746319, 'nDCG@500': 0.6768959647957119, 'nDCG@1000': 0.6918869483679339, 'R(rel=2)@3': 0.16003239577496967, 'R(rel=2)@5': 0.2476104839708578, 'R(rel=2)@10': 0.38692970913678476, 'R(rel=2)@25': 0.5416199547444053, 'R(rel=2)@50': 0.6528313321662219, 'R(rel=2)@100': 0.7314423202243299, 'R(rel=2)@200': 0.7832381221272239, 'R(rel=2)@1000': 0.853532365555859, 'P(rel=2)@1': 0.7037037037037037, 'P(rel=2)@3': 0.6543209876543209, 'P(rel=2)@5': 0.6111111111111113, 'P(rel=2)@10': 0.5296296296296296, 'P(rel=2)@25': 0.36444444444444457, 'P(rel=2)@50': 0.2514814814814814, 'P(rel=2)@100': 0.157037037037037, 'P(rel=2)@200': 0.08990740740740742, 'P(rel=2)@1000': 0.021296296296296303, 'AP(rel=2)@1000': 0.4660034706836442, 'RR(rel=2)@10': 0.8140432098765432, 'NumRet': 54000.0, 'num_q': 54.0}


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

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
   - T2: ColBERT-RF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"

colbert-25000.dnn : MRR@10 0.36271177286578404 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/MSMARCO-psg/test.py/2021-11-05_07.33.47/ranking.tsv)
colbert-50000.dnn : MRR@10 0.3656893846363755 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/MSMARCO-psg/test.py/2021-11-05_07.33.54/ranking.tsv)
colbert-**100000**.dnn: MRR@10 **0.36811462455087024** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/MSMARCO-psg/test.py/2021-11-06_05.56.54/ranking.tsv)
colbert-150000.dnn: MRR@10 0.3659496520671309 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/MSMARCO-psg/test.py/2021-11-06_05.57.00/ranking.tsv)
colbert-200000.dnn: MRR@10 0.3647866352844855 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/MSMARCO-psg/test.py/2021-11-06_05.57.05/ranking.tsv)

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
   - T2: ColBERT-PRF (beta 1.0) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt"

colbert-50000.dnn : MRR@10 0.36552337745031105 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/MSMARCO-psg/test.py/2021-11-06_18.14.11/ranking.tsv) # dilab4
colbert-**100000**.dnn: MRR@10 **0.36853771546823094** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/MSMARCO-psg/test.py/2021-11-06_16.20.48/ranking.tsv) # dilab4
colbert-150000.dnn: MRR@10 0.36819688907081394 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/MSMARCO-psg/test.py/2021-11-06_16.21.00/ranking.tsv) # dilab4
colbert-200000.dnn: MRR@10 0.3661451653249638 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/MSMARCO-psg/test.py/2021-11-06_18.14.25/ranking.tsv) # dilab4

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
   - T2: ColBERT-PRF (docT5query, beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt"

colbert-50000.dnn : MRR@10 0.36578904579979094 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_18.18.59/ranking.tsv) # dilab4
colbert-100000.dnn: MRR@10 0.3696736708054758 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_18.19.11/ranking.tsv) # dilab4
colbert-**150000**.dnn: MRR@10 **0.3713696161368083** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_18.19.21/ranking.tsv) # dilab4
colbert-200000.dnn: MRR@10 0.36860224223404714 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_18.19.33/ranking.tsv) # dilab4

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 1.0) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt"
   - T2: ColBERT-PRF (docT5query, beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt"

colbert-50000.dnn : MRR@10 0.3662259516987307 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_09.41.31/ranking.tsv) # dilab003
colbert-100000.dnn: MRR@10 0.36814788283985983 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_09.41.39/ranking.tsv) # dilab003
colbert-**150000**.dnn: MRR@10 **0.36995355209896713** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_18.30.39/ranking.tsv) # sonic
colbert-200000.dnn: MRR@10 0.3689101514531307 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_09.41.44/ranking.tsv) # dilab003

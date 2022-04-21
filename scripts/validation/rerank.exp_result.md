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

## Ours. mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT (experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) instead of RF
- Updating expansion embeddings step-wise for every 50k iterations, for mutual learning between the teacher and the student

colbert-25000.dnn : MRR@10 0.36042360258334444 (experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py/2021-11-02_06.58.17)
colbert-50000.dnn : MRR@10 0.3667617228362205 (experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py/2021-11-02_08.56.10)

## Ours. mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT (experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) instead of RF
- Updating expansion embeddings step-wise for every 50k iterations, for mutual learning between the teacher and the student

colbert-75000.dnn  : MRR@10 0.36589388047482607 (experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg/test.py/2021-11-03_06.28.18)
colbert-100000.dnn : MRR@10 gpu1 (experiments/mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.100k/MSMARCO-psg/test.py/2021-11-03_09.30.04)

## Ours. mutual_learning/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.150k
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT (experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) instead of RF
- Updating expansion embeddings step-wise for every 50k iterations, for mutual learning between the teacher and the student

colbert-125000.dnn : MRR@10  ()
colbert-150000.dnn : MRR@10  ()




# DEPRECATED!
## Ours. mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT (experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl)
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) instead of RF
- Updating expansion embeddings step-wise for every 50k iterations, for mutual learning between the teacher and the student
- KD objective = Margin-MSE (with max margin threshold = 32.0; to match the maximum length of a query)

colbert-25000.dnn : MRR@10 gpu0 (experiments/mutual_learning-margin_mse/finetuned.b36.lr3e6.hn.kd_prf.beta0.5.50k/MSMARCO-psg/test.py/2021-11-02_10.24.17)
colbert-50000.dnn : MRR@10  ()


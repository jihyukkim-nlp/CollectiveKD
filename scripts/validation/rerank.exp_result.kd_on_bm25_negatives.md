# Baseline: ColBERT (copied from paper https://arxiv.org/pdf/2004.12832.pdf)
Re-ranking
   MRR@10: 0.349
End-to-end ranking
   MRR@10 = 0.360, Recall@50 = 0.829, Recall@200 = 0.923, Recall@1000 = 0.968

# Baseline: ColBERT (our implementation - trained using in-batch negatives) (colbert.teacher.dnn)
Re-ranking (300k iterations)
   MRR@10: 0.354
End-to-end ranking
   MRR@10 = 0.367, Recall@50 = 0.833, Recall@200 = 0.925, Recall@1000 = 0.967

TREC 2019 
{'nDCG@10': 0.6996191676961366, 'nDCG@25': 0.6611577340895359, 'nDCG@50': 0.6489031290874794, 'nDCG@100': 0.6445046665392238, 'nDCG@200': 0.6575805267886845, 'nDCG@500': 0.6897259199696175, 'nDCG@1000': 0.7122704288228963, 'R(rel=2)@3': 0.13427849571086078, 'R(rel=2)@5': 0.1841068344211011, 'R(rel=2)@10': 0.2875825212503319, 'R(rel=2)@25': 0.41898498969474013, 'R(rel=2)@50': 0.533461722369848, 'R(rel=2)@100': 0.6350332824018848, 'R(rel=2)@200': 0.7204932770174376, 'R(rel=2)@1000': 0.8370813777761269, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.7286821705426357, 'P(rel=2)@5': 0.7069767441860467, 'P(rel=2)@10': 0.6325581395348838, 'P(rel=2)@25': 0.46790697674418597, 'P(rel=2)@50': 0.3572093023255815, 'P(rel=2)@100': 0.2590697674418605, 'P(rel=2)@200': 0.16999999999999998, 'P(rel=2)@1000': 0.043837209302325555, 'AP(rel=2)@1000': 0.4753639977588095, 'RR(rel=2)@10': 0.8330103359173127, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020 
{'nDCG@10': 0.674625300218906, 'nDCG@25': 0.6404223915640629, 'nDCG@50': 0.628640636810472, 'nDCG@100': 0.6311473087514119, 'nDCG@200': 0.6521529878746319, 'nDCG@500': 0.6768959647957119, 'nDCG@1000': 0.6918869483679339, 'R(rel=2)@3': 0.16003239577496967, 'R(rel=2)@5': 0.2476104839708578, 'R(rel=2)@10': 0.38692970913678476, 'R(rel=2)@25': 0.5416199547444053, 'R(rel=2)@50': 0.6528313321662219, 'R(rel=2)@100': 0.7314423202243299, 'R(rel=2)@200': 0.7832381221272239, 'R(rel=2)@1000': 0.853532365555859, 'P(rel=2)@1': 0.7037037037037037, 'P(rel=2)@3': 0.6543209876543209, 'P(rel=2)@5': 0.6111111111111113, 'P(rel=2)@10': 0.5296296296296296, 'P(rel=2)@25': 0.36444444444444457, 'P(rel=2)@50': 0.2514814814814814, 'P(rel=2)@100': 0.157037037037037, 'P(rel=2)@200': 0.08990740740740742, 'P(rel=2)@1000': 0.021296296296296303, 'AP(rel=2)@1000': 0.4660034706836442, 'RR(rel=2)@10': 0.8140432098765432, 'NumRet': 54000.0, 'num_q': 54.0}


# Newly Experimented (Fine-tuning ColBERT)

## Ours. kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n
- Training ColBERT using labeled positive and BM25 negatives
   - Train triple: "data/triples.train.small.ids.jsonl"
   - Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
      - kd_expansion_pt (PRF): "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"

Validation performance (on re-reanking task)
colbert-25000.dnn : MRR@10 0.33041018556419643 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-05_01.30.07/ranking.tsv)
colbert-50000.dnn : MRR@10 0.34268408605084816 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-05_04.07.44/ranking.tsv)
colbert-75000.dnn : MRR@10 0.34468993041342644 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-05_06.55.58/ranking.tsv)
colbert-100000.dnn: MRR@10 0.34806186610269657 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-05_09.39.43/ranking.tsv)
colbert-150000.dnn: MRR@10 0.3483370855505531 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-05_07.13.00/ranking.tsv)
colbert-200000.dnn: MRR@10 0.351041751944331 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-06_01.06.31/ranking.tsv)
colbert-250000.dnn: MRR@10 0.3517467139673448 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-06_01.07.19/ranking.tsv)
colbert-300000.dnn: MRR@10 0.3523408718788375 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-06_01.09.23/ranking.tsv)
colbert-350000.dnn: MRR@10 0.35312258379951766 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-06_18.27.53/ranking.tsv)
colbert-400000.dnn: MRR@10 0.35356921135216257 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-06_23.24.51/ranking.tsv) # sonic
<!-- resume training: --resume --resume_optimizer --checkpoint -->
colbert-450000.dnn: MRR@10 0.35198145495065275 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/2021-11-07_19.39.56/ranking.tsv) # sonic
colbert-500000.dnn: MRR@10 0.35238663755855726 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/2021-11-07_19.40.09/ranking.tsv) # sonic
colbert-550000.dnn: MRR@10 0.355203756765361 (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/2021-11-07_19.40.22/ranking.tsv) # sonic
colbert-**600000**.dnn: MRR@10 **0.3555993882748899** (experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/2021-11-08_01.00.45/ranking.tsv) # sonic

## Ours. kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n
- Training ColBERT using labeled positive and BM25 negatives
   - Train triple: "data/triples.train.small.ids.jsonl"
   - Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
      - kd_expansion_pt (PRF): "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt"

Validation performance (on re-reanking task)
colbert-100000.dnn: MRR@10 0.3452139332332741 (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-07_19.41.27/ranking.tsv)
colbert-200000.dnn: MRR@10 0.3513765520534862 (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-08_05.53.05/ranking.tsv)
colbert-300000.dnn: MRR@10 0.35267885568745094 (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-08_20.52.35/ranking.tsv)
colbert-400000.dnn: MRR@10 0.35351770364306107 (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-09_02.27.07/ranking.tsv)
colbert-450000.dnn: MRR@10 **0.3537730818210763** (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-09_23.18.22/ranking.tsv)
colbert-500000.dnn: MRR@10 0.3522831673261471 (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-09_23.17.41/ranking.tsv)
colbert-550000.dnn: MRR@10 0.3529164392595624 (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-10_02.22.42/ranking.tsv)
colbert-600000.dnn: MRR@10 0.3517901487242457 (experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/2021-11-10_02.22.51/ranking.tsv)
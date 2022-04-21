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
{'nDCG@10': 0.7414371749888692, 'nDCG@25': 0.6912861283638474, 'nDCG@50': 0.6651141257207117, 'nDCG@100': 0.6546823830897742, 'nDCG@200': 0.6682710830890339, 'nDCG@500': 0.7006364918196664, 'nDCG@1000': 0.717662830416884, 'R(rel=2)@3': 0.14006075962131528, 'R(rel=2)@5': 0.18914642611621504, 'R(rel=2)@10': 0.2894266186254735, 'R(rel=2)@25': 0.43227938979752983, 'R(rel=2)@50': 0.5325804141762943, 'R(rel=2)@100': 0.6262149523371159, 'R(rel=2)@200': 0.7176619981791489, 'R(rel=2)@1000': 0.8378453338575937, 'P(rel=2)@3': 0.7751937984496124, 'P(rel=2)@5': 0.7209302325581395, 'P(rel=2)@10': 0.6441860465116279, 'P(rel=2)@25': 0.49209302325581405, 'P(rel=2)@50': 0.36511627906976757, 'P(rel=2)@100': 0.2653488372093023, 'P(rel=2)@200': 0.17302325581395347, 'AP(rel=2)@1000': 0.48508580264653356, 'RR(rel=2)@10': 0.9031007751937985, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/rerank.py/2021-10-12_14.24.58/ranking.tsv)
{'nDCG@10': 0.6998388784820837, 'nDCG@25': 0.6701788797695977, 'nDCG@50': 0.6520576880566012, 'nDCG@100': 0.6495910621986439, 'nDCG@200': 0.6762072849837231, 'nDCG@500': 0.7025032136354434, 'nDCG@1000': 0.7159782428303274, 'R(rel=2)@3': 0.1754122377184954, 'R(rel=2)@5': 0.2592727410388973, 'R(rel=2)@10': 0.3869865168153549, 'R(rel=2)@25': 0.5604024403846639, 'R(rel=2)@50': 0.6675922731394245, 'R(rel=2)@100': 0.7405625934528436, 'R(rel=2)@200': 0.7985698819377204, 'R(rel=2)@1000': 0.8670388401182964, 'P(rel=2)@3': 0.7098765432098765, 'P(rel=2)@5': 0.6481481481481484, 'P(rel=2)@10': 0.5425925925925926, 'P(rel=2)@25': 0.38074074074074066, 'P(rel=2)@50': 0.2637037037037037, 'P(rel=2)@100': 0.16407407407407407, 'P(rel=2)@200': 0.09490740740740743, 'AP(rel=2)@1000': 0.49370822262673547, 'RR(rel=2)@10': 0.8370370370370371, 'NumRet': 54000.0, 'num_q': 54.0}

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.k10.beta1.0.clusters10/label.py/2021-10-17_01.17.56/expansion.pt)

Validation performance (on re-reanking task)
colbert-50000.dnn : MRR@10 **0.36324219993632545** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans/MSMARCO-psg/test.py/2021-10-24_13.02.26)

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt)

Validation performance (on re-reanking task)
colbert-**150000**.dnn: MRR@10 **0.3650446854959745** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/MSMARCO-psg/test.py/2021-10-28_05.09.11)

End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/MSMARCO-psg/rerank.py/2021-10-28_19.11.32/ranking.tsv)
[Oct 28, 19:21:54] #> MRR@10 = 0.3783029176331472
[Oct 28, 19:21:54] #> MRR@100 = 0.3886694256720538
[Oct 28, 19:21:54] #> Recall@50 = 0.8457975167144222
[Oct 28, 19:21:54] #> Recall@200 = 0.9267908309455589
[Oct 28, 19:21:54] #> Recall@1000 = 0.9684455587392552

TREC 2019: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/TREC2019-psg/rerank.py/2021-10-28_19.08.45/ranking.tsv)
{'nDCG@10': 0.7281225466174466, 'nDCG@25': 0.6815457834165888, 'nDCG@50': 0.6527105258963833, 'nDCG@100': 0.6416597721303212, 'nDCG@200': 0.6535450695913663, 'nDCG@500': 0.6841751490432233, 'nDCG@1000': 0.7006268056820996, 'R(rel=2)@3': 0.14101791337185784, 'R(rel=2)@5': 0.1891983338702567, 'R(rel=2)@10': 0.2823092727024237, 'R(rel=2)@25': 0.42851097054827564, 'R(rel=2)@50': 0.5248671965087449, 'R(rel=2)@100': 0.6283601709022288, 'R(rel=2)@200': 0.7107401702277394, 'R(rel=2)@1000': 0.816438363117514, 'P(rel=2)@3': 0.75968992248062, 'P(rel=2)@5': 0.7162790697674418, 'P(rel=2)@10': 0.6395348837209304, 'P(rel=2)@25': 0.4855813953488374, 'P(rel=2)@50': 0.3586046511627907, 'P(rel=2)@100': 0.25395348837209303, 'P(rel=2)@200': 0.16441860465116276, 'AP(rel=2)@1000': 0.4716678095768029, 'RR(rel=2)@10': 0.8798449612403101, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_rf_beta0.5/TREC2020-psg/rerank.py/2021-10-28_19.08.56/ranking.tsv)
{'nDCG@10': 0.6941581102769994, 'nDCG@25': 0.6621584688568556, 'nDCG@50': 0.6400189469223067, 'nDCG@100': 0.6372299977246925, 'nDCG@200': 0.6609327838388461, 'nDCG@500': 0.6909222322197158, 'nDCG@1000': 0.7058789656315445, 'R(rel=2)@3': 0.15768513626681552, 'R(rel=2)@5': 0.26242399678543804, 'R(rel=2)@10': 0.40849449828837786, 'R(rel=2)@25': 0.5621631069236683, 'R(rel=2)@50': 0.6606143673097099, 'R(rel=2)@100': 0.7372899258925325, 'R(rel=2)@200': 0.7921012000713794, 'R(rel=2)@1000': 0.8656621658642449, 'P(rel=2)@3': 0.6604938271604938, 'P(rel=2)@5': 0.637037037037037, 'P(rel=2)@10': 0.5537037037037037, 'P(rel=2)@25': 0.3844444444444444, 'P(rel=2)@50': 0.2607407407407408, 'P(rel=2)@100': 0.16370370370370374, 'P(rel=2)@200': 0.09351851851851853, 'AP(rel=2)@1000': 0.4871058510463613, 'RR(rel=2)@10': 0.8379629629629629, 'NumRet': 54000.0, 'num_q': 54.0}

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF instead of RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt)

Validation performance (on re-reanking task)
colbert-**150000**.dnn: MRR@10 **0.36830644244326183** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/test.py/2021-10-25_06.33.55)

End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/rerank.py/2021-10-27_18.16.49/ranking.tsv)
[Oct 27, 18:25:30] #> MRR@10 = 0.3838565970800934
[Oct 27, 18:25:30] #> MRR@100 = 0.394314653556914
[Oct 27, 18:25:30] #> Recall@50 = 0.855276981852913
[Oct 27, 18:25:30] #> Recall@200 = 0.9339660936007642
[Oct 27, 18:25:30] #> Recall@1000 = 0.9735912129894939

TREC 2019: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf//TREC2019-psg/rerank.py/2021-10-27_18.27.16/ranking.tsv)
{'nDCG@10': 0.751479264408955, 'nDCG@25': 0.7068279534414165, 'nDCG@50': 0.6865507085592907, 'nDCG@100': 0.6693198864645404, 'nDCG@200': 0.6839463012454996, 'nDCG@500': 0.7175722912626523, 'nDCG@1000': 0.738852586046282, 'R(rel=2)@3': 0.13980394357142115, 'R(rel=2)@5': 0.19573292356075234, 'R(rel=2)@10': 0.29480750771231007, 'R(rel=2)@25': 0.43983537941528333, 'R(rel=2)@50': 0.5481105754747899, 'R(rel=2)@100': 0.6456108163296869, 'R(rel=2)@200': 0.7420251450402223, 'R(rel=2)@1000': 0.8682538608133842, 'P(rel=2)@3': 0.7751937984496124, 'P(rel=2)@5': 0.7488372093023257, 'P(rel=2)@10': 0.6674418604651163, 'P(rel=2)@25': 0.4986046511627908, 'P(rel=2)@50': 0.38186046511627914, 'P(rel=2)@100': 0.27139534883720934, 'P(rel=2)@200': 0.17837209302325582, 'AP(rel=2)@1000': 0.4990087498023621, 'RR(rel=2)@10': 0.8875968992248063, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf//TREC2020-psg/rerank.py/2021-10-27_18.32.10/ranking.tsv)
{'nDCG@10': 0.7212992564750679, 'nDCG@25': 0.6882333570303786, 'nDCG@50': 0.6710105897744095, 'nDCG@100': 0.6643247389738698, 'nDCG@200': 0.6900428467955689, 'nDCG@500': 0.7201765609094679, 'nDCG@1000': 0.7347458071323215, 'R(rel=2)@3': 0.17594059510198454, 'R(rel=2)@5': 0.2781593036544907, 'R(rel=2)@10': 0.4082851513139375, 'R(rel=2)@25': 0.5684508031410702, 'R(rel=2)@50': 0.673453780994433, 'R(rel=2)@100': 0.7474984290820841, 'R(rel=2)@200': 0.80802478471011, 'R(rel=2)@1000': 0.8831295923129652, 'P(rel=2)@3': 0.7160493827160493, 'P(rel=2)@5': 0.6703703703703705, 'P(rel=2)@10': 0.5648148148148148, 'P(rel=2)@25': 0.39185185185185173, 'P(rel=2)@50': 0.27, 'P(rel=2)@100': 0.1666666666666667, 'P(rel=2)@200': 0.09583333333333333, 'AP(rel=2)@1000': 0.5091864167343898, 'RR(rel=2)@10': 0.8904320987654322, 'NumRet': 54000.0, 'num_q': 54.0}

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 1.0) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt"
   - T2: ColBERT-PRF (docT5query, beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt"

Validation performance (on re-reanking task)
colbert-**150000**.dnn: MRR@10 **0.36995355209896713** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_18.30.39/ranking.tsv) # sonic

End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/MSMARCO-psg/rerank.py/2021-11-07_05.50.27/ranking.tsv)
[Nov 07, 05:58:26] #> MRR@10 = 0.3867435302678867
[Nov 07, 05:58:26] #> MRR@100 = 0.3968163278163295
[Nov 07, 05:58:26] #> Recall@50 = 0.8557067812798471
[Nov 07, 05:58:26] #> Recall@200 = 0.9356136580706782
[Nov 07, 05:58:26] #> Recall@1000 = 0.9726599808978034

TREC 2019: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/TREC2019-psg/rerank.py/2021-11-07_06.00.20/ranking.tsv)
{'nDCG@10': 0.747915276431666, 'nDCG@25': 0.7113580426122609, 'nDCG@50': 0.6884556532676476, 'nDCG@100': 0.6670425986943763, 'nDCG@200': 0.6798770950747327, 'nDCG@500': 0.7159702402420008, 'nDCG@1000': 0.735375827948725, 'R(rel=2)@3': 0.13912615032414405, 'R(rel=2)@5': 0.18886692134591287, 'R(rel=2)@10': 0.2844897937961738, 'R(rel=2)@25': 0.43599646329864783, 'R(rel=2)@50': 0.5580377678184294, 'R(rel=2)@100': 0.6334425956530776, 'R(rel=2)@200': 0.7277127656446579, 'R(rel=2)@1000': 0.8554473979135052, 'P(rel=2)@1': 0.7906976744186046, 'P(rel=2)@3': 0.7829457364341085, 'P(rel=2)@5': 0.7302325581395349, 'P(rel=2)@10': 0.6581395348837209, 'P(rel=2)@25': 0.5032558139534884, 'P(rel=2)@50': 0.38325581395348846, 'P(rel=2)@100': 0.26651162790697674, 'P(rel=2)@200': 0.1752325581395349, 'P(rel=2)@1000': 0.0449302325581395, 'AP(rel=2)@1000': 0.4992260562272171, 'RR(rel=2)@10': 0.8792912513842747, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta1.0.t2.prf.docT5query.beta0.5/TREC2020-psg/rerank.py/2021-11-07_06.05.17/ranking.tsv)
{'nDCG@10': 0.7261155200816851, 'nDCG@25': 0.6872333285376632, 'nDCG@50': 0.6743072359932581, 'nDCG@100': 0.6682313148683671, 'nDCG@200': 0.6914641842706998, 'nDCG@500': 0.7219790044261803, 'nDCG@1000': 0.7362660917546165, 'R(rel=2)@3': 0.18708551911554902, 'R(rel=2)@5': 0.28614473688414777, 'R(rel=2)@10': 0.41172077286338565, 'R(rel=2)@25': 0.5709202956008289, 'R(rel=2)@50': 0.67462364030886, 'R(rel=2)@100': 0.7429768678077224, 'R(rel=2)@200': 0.7957556102197711, 'R(rel=2)@1000': 0.8769805401465672, 'P(rel=2)@1': 0.8148148148148148, 'P(rel=2)@3': 0.7592592592592593, 'P(rel=2)@5': 0.6962962962962963, 'P(rel=2)@10': 0.5648148148148148, 'P(rel=2)@25': 0.3881481481481479, 'P(rel=2)@50': 0.26851851851851855, 'P(rel=2)@100': 0.16481481481481483, 'P(rel=2)@200': 0.09444444444444444, 'P(rel=2)@1000': 0.022574074074074076, 'AP(rel=2)@1000': 0.5112083641367554, 'RR(rel=2)@10': 0.8842592592592593, 'NumRet': 54000.0, 'num_q': 54.0}

## Ours. finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Different expansion embeddings & term selection method, for query expansion, from the ``finetuned.b36.lr3e6.hn.kd_qe``: QDMaxSim -> KMeans
   - Using PRF (beta 0.5) instead of RF (experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt)

Validation performance (on re-reanking task)
colbert-**150000**.dnn: MRR@10 **0.3704569177241098** (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/test.py/2021-10-26_04.18.23)

End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/MSMARCO-psg/rerank.py/2021-10-28_17.14.12/ranking.tsv)
[Oct 28, 17:22:18] #> MRR@10 = 0.38573770864601875
[Oct 28, 17:22:18] #> MRR@100 = 0.39612999178929736
[Oct 28, 17:22:18] #> Recall@50 = 0.855276981852913
[Oct 28, 17:22:18] #> Recall@200 = 0.9365329512893984
[Oct 28, 17:22:18] #> Recall@1000 = 0.9724808978032475

TREC 2019: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/TREC2019-psg/rerank.py/2021-10-28_17.09.53/ranking.tsv)
{'nDCG@10': 0.7444657154258438, 'nDCG@25': 0.700609112467417, 'nDCG@50': 0.6782362779967904, 'nDCG@100': 0.6625906052930649, 'nDCG@200': 0.67660462384775, 'nDCG@500': 0.7081542035478159, 'nDCG@1000': 0.7284524224158535, 'R(rel=2)@3': 0.13721996424325578, 'R(rel=2)@5': 0.19419671633821498, 'R(rel=2)@10': 0.2867178702758005, 'R(rel=2)@25': 0.43487560616983434, 'R(rel=2)@50': 0.5416819225785917, 'R(rel=2)@100': 0.6317803173067645, 'R(rel=2)@200': 0.7262171594511034, 'R(rel=2)@1000': 0.8431706927747813, 'P(rel=2)@3': 0.7674418604651163, 'P(rel=2)@5': 0.7395348837209302, 'P(rel=2)@10': 0.6558139534883721, 'P(rel=2)@25': 0.4948837209302326, 'P(rel=2)@50': 0.3767441860465117, 'P(rel=2)@100': 0.2644186046511628, 'P(rel=2)@200': 0.17395348837209298, 'AP(rel=2)@1000': 0.4917772900026287, 'RR(rel=2)@10': 0.8992248062015503, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/TREC2020-psg/rerank.py/2021-10-28_17.09.50/ranking.tsv)
{'nDCG@10': 0.7242514449292434, 'nDCG@25': 0.6874271292815659, 'nDCG@50': 0.6689929685326551, 'nDCG@100': 0.6634512390168448, 'nDCG@200': 0.6842849149440485, 'nDCG@500': 0.7162029357361335, 'nDCG@1000': 0.73017988905081, 'R(rel=2)@3': 0.1774525556830816, 'R(rel=2)@5': 0.2952743935120175, 'R(rel=2)@10': 0.40503055955032585, 'R(rel=2)@25': 0.5728708073062028, 'R(rel=2)@50': 0.6720374350007257, 'R(rel=2)@100': 0.7425916514932751, 'R(rel=2)@200': 0.7877326007000796, 'R(rel=2)@1000': 0.8738134872402924, 'P(rel=2)@3': 0.7345679012345678, 'P(rel=2)@5': 0.7037037037037038, 'P(rel=2)@10': 0.5611111111111111, 'P(rel=2)@25': 0.3866666666666665, 'P(rel=2)@50': 0.2633333333333333, 'P(rel=2)@100': 0.16592592592592595, 'P(rel=2)@200': 0.09342592592592594, 'AP(rel=2)@1000': 0.508362088048785, 'RR(rel=2)@10': 0.8996913580246914, 'NumRet': 54000.0, 'num_q': 54.0}

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
   - T2: ColBERT-PRF (beta 1.0) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt"

Validation performance (on re-reanking task)
colbert-**100000**.dnn: MRR@10 **0.36853771546823094** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/MSMARCO-psg/test.py/2021-11-06_16.20.48/ranking.tsv) # dilab4
colbert-150000.dnn: MRR@10 0.36819688907081394 (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/MSMARCO-psg/test.py/2021-11-06_16.21.00/ranking.tsv) # dilab4

End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/MSMARCO-psg/rerank.py/2021-11-07_15.20.52/ranking.tsv)
[Nov 07, 15:28:53] #> MRR@10 = 0.38428571428571445
[Nov 07, 15:28:53] #> MRR@100 = 0.39465592876657507
[Nov 07, 15:28:53] #> Recall@50 = 0.8575811843361987
[Nov 07, 15:28:53] #> Recall@200 = 0.9351838586437441
[Nov 07, 15:28:53] #> Recall@1000 = 0.9743075453677174

TREC 2019: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/TREC2019-psg/rerank.py/2021-11-07_17.16.26/ranking.tsv)
{'nDCG@10': 0.7447804788523331, 'nDCG@25': 0.706178366857354, 'nDCG@50': 0.6804322607495185, 'nDCG@100': 0.6652837449605709, 'nDCG@200': 0.677420863578689, 'nDCG@500': 0.7118000111586239, 'nDCG@1000': 0.7350766367701019, 'R(rel=2)@3': 0.14009514257220607, 'R(rel=2)@5': 0.19261796799259198, 'R(rel=2)@10': 0.2902918619036477, 'R(rel=2)@25': 0.4443515606607393, 'R(rel=2)@50': 0.538429300385793, 'R(rel=2)@100': 0.6380317097727511, 'R(rel=2)@200': 0.7269199044606026, 'R(rel=2)@1000': 0.8639822864020344, 'P(rel=2)@1': 0.7674418604651163, 'P(rel=2)@3': 0.7906976744186046, 'P(rel=2)@5': 0.7488372093023254, 'P(rel=2)@10': 0.6604651162790697, 'P(rel=2)@25': 0.5051162790697675, 'P(rel=2)@50': 0.3804651162790699, 'P(rel=2)@100': 0.27093023255813964, 'P(rel=2)@200': 0.17674418604651165, 'P(rel=2)@1000': 0.04541860465116277, 'AP(rel=2)@1000': 0.49436371091812875, 'RR(rel=2)@10': 0.872093023255814, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.beta1.0/TREC2020-psg/rerank.py/2021-11-07_18.15.09/ranking.tsv)
{'nDCG@10': 0.7221564177862964, 'nDCG@25': 0.6910264947589114, 'nDCG@50': 0.6725049208600912, 'nDCG@100': 0.6660728261680683, 'nDCG@200': 0.6912879372234738, 'nDCG@500': 0.7214689882633865, 'nDCG@1000': 0.7356084168046826, 'R(rel=2)@3': 0.18153231430829475, 'R(rel=2)@5': 0.28239830892777384, 'R(rel=2)@10': 0.40379329370290334, 'R(rel=2)@25': 0.57732709626984, 'R(rel=2)@50': 0.6751188325769218, 'R(rel=2)@100': 0.7467889891337471, 'R(rel=2)@200': 0.8023222638644799, 'R(rel=2)@1000': 0.8808790067275595, 'P(rel=2)@1': 0.8148148148148148, 'P(rel=2)@3': 0.7345679012345678, 'P(rel=2)@5': 0.6814814814814816, 'P(rel=2)@10': 0.5592592592592592, 'P(rel=2)@25': 0.39259259259259244, 'P(rel=2)@50': 0.27148148148148143, 'P(rel=2)@100': 0.16722222222222227, 'P(rel=2)@200': 0.09629629629629628, 'P(rel=2)@1000': 0.02272222222222222, 'AP(rel=2)@1000': 0.5104444424591335, 'RR(rel=2)@10': 0.8873456790123456, 'NumRet': 54000.0, 'num_q': 54.0}

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
   - T2: ColBERT-PRF (docT5query, beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-prf.fb_docT5query.docs3.k10.beta0.5.clusters24/label.py/2021-11-05_10.35.11/expansion.pt"

Validation performance (on re-reanking task)
colbert-**150000**.dnn: MRR@10 **0.3713696161368083** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/test.py/2021-11-06_18.19.21/ranking.tsv) # dilab4

End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/MSMARCO-psg/rerank.py/2021-11-07_04.36.56/ranking.tsv)
[Nov 07, 04:45:06] #> MRR@10 = 0.3876398553690825
[Nov 07, 04:45:06] #> MRR@100 = 0.39756275683337217
[Nov 07, 04:45:06] #> Recall@50 = 0.8572827125119389
[Nov 07, 04:45:06] #> Recall@200 = 0.9374522445081186
[Nov 07, 04:45:06] #> Recall@1000 = 0.9741284622731617

TREC 2019: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/TREC2019-psg/rerank.py/2021-11-07_04.24.09/ranking.tsv)
{'nDCG@10': 0.7477697118514163, 'nDCG@25': 0.7045798296168276, 'nDCG@50': 0.6817599129259879, 'nDCG@100': 0.6658601810601122, 'nDCG@200': 0.6760475664283441, 'nDCG@500': 0.7119134978683709, 'nDCG@1000': 0.7318251201708097, 'R(rel=2)@3': 0.14074877393485613, 'R(rel=2)@5': 0.1928194029728131, 'R(rel=2)@10': 0.2849414766227611, 'R(rel=2)@25': 0.43208048471856836, 'R(rel=2)@50': 0.547926322370301, 'R(rel=2)@100': 0.6362810114410614, 'R(rel=2)@200': 0.7264601361425026, 'R(rel=2)@1000': 0.8483177488021568, 'P(rel=2)@1': 0.813953488372093, 'P(rel=2)@3': 0.7829457364341088, 'P(rel=2)@5': 0.7488372093023257, 'P(rel=2)@10': 0.6581395348837209, 'P(rel=2)@25': 0.4958139534883721, 'P(rel=2)@50': 0.37906976744186055, 'P(rel=2)@100': 0.26837209302325593, 'P(rel=2)@200': 0.17488372093023255, 'P(rel=2)@1000': 0.04479069767441858, 'AP(rel=2)@1000': 0.49445202195926663, 'RR(rel=2)@10': 0.8934108527131782, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.prf.docT5query.beta0.5/TREC2020-psg/rerank.py/2021-11-07_04.29.14/ranking.tsv)
{'nDCG@10': 0.7219150298265465, 'nDCG@25': 0.6876136982953263, 'nDCG@50': 0.6694310906118555, 'nDCG@100': 0.6661066377146191, 'nDCG@200': 0.6867651679276188, 'nDCG@500': 0.719885499579943, 'nDCG@1000': 0.7337688685666653, 'R(rel=2)@3': 0.18103433733972954, 'R(rel=2)@5': 0.2841274674304856, 'R(rel=2)@10': 0.4052529490388513, 'R(rel=2)@25': 0.5716865239167432, 'R(rel=2)@50': 0.6750581812208888, 'R(rel=2)@100': 0.7510559531868857, 'R(rel=2)@200': 0.7917652548259921, 'R(rel=2)@1000': 0.8786416721213636, 'P(rel=2)@1': 0.8148148148148148, 'P(rel=2)@3': 0.7407407407407407, 'P(rel=2)@5': 0.6851851851851852, 'P(rel=2)@10': 0.5574074074074075, 'P(rel=2)@25': 0.3896296296296295, 'P(rel=2)@50': 0.2674074074074074, 'P(rel=2)@100': 0.1675925925925926, 'P(rel=2)@200': 0.0947222222222222, 'P(rel=2)@1000': 0.0227037037037037, 'AP(rel=2)@1000': 0.5102749112864325, 'RR(rel=2)@10': 0.8873456790123456, 'NumRet': 54000.0, 'num_q': 54.0}

## Ours. ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5
- Training using new train triples: 
   - HN (Hard Negatives) from pre-trained ColBERT 
   - triple: "experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl"
- Training using KD (Knowledge Distillation) from pre-trained ColBERT using expanded query
   - Using ensemble of teachers
   - T1: ColBERT-PRF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
   - T2: ColBERT-RF (beta 0.5) "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.rf.k10.beta0.5.clusters10/label.py/expansion.pt"

Validation performance (on re-reanking task)
colbert-**100000**.dnn: MRR@10 **0.36811462455087024** (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/MSMARCO-psg/test.py/2021-11-06_05.56.54/ranking.tsv)

End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/MSMARCO-psg/rerank.py/2021-11-07_01.11.09/ranking.tsv)
[Nov 07, 03:15:58] #> MRR@10 = 0.3831523172783918
[Nov 07, 03:15:58] #> MRR@100 = 0.3930414287068396
[Nov 07, 03:15:58] #> Recall@50 = 0.8512416427889208
[Nov 07, 03:15:58] #> Recall@200 = 0.9343361986628462
[Nov 07, 03:15:58] #> Recall@1000 = 0.9718123209169056

TREC 2019: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/TREC2019-psg/rerank.py/2021-11-07_01.04.41/ranking.tsv)
{'nDCG@10': 0.7433041544013037, 'nDCG@25': 0.6985305411634725, 'nDCG@50': 0.6694597472367975, 'nDCG@100': 0.6537756523567937, 'nDCG@200': 0.6662849256167669, 'nDCG@500': 0.6972279782041914, 'nDCG@1000': 0.7184552816978748, 'R(rel=2)@3': 0.14299615628717327, 'R(rel=2)@5': 0.1905682347800681, 'R(rel=2)@10': 0.29208393265670685, 'R(rel=2)@25': 0.44019837370299564, 'R(rel=2)@50': 0.5379928809915712, 'R(rel=2)@100': 0.6320711613170237, 'R(rel=2)@200': 0.714561768848796, 'R(rel=2)@1000': 0.8353133536088521, 'P(rel=2)@1': 0.8372093023255814, 'P(rel=2)@3': 0.7751937984496127, 'P(rel=2)@5': 0.7395348837209303, 'P(rel=2)@10': 0.6534883720930234, 'P(rel=2)@25': 0.4976744186046512, 'P(rel=2)@50': 0.3632558139534884, 'P(rel=2)@100': 0.2586046511627907, 'P(rel=2)@200': 0.16779069767441857, 'P(rel=2)@1000': 0.04241860465116276, 'AP(rel=2)@1000': 0.4881129694714285, 'RR(rel=2)@10': 0.9116279069767442, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/ensemble/finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5/TREC2020-psg/rerank.py/2021-11-07_03.19.30/ranking.tsv)
{'nDCG@10': 0.7060198098666223, 'nDCG@25': 0.6774981999127534, 'nDCG@50': 0.6594566755034486, 'nDCG@100': 0.6550557275003618, 'nDCG@200': 0.6770069658534784, 'nDCG@500': 0.7080157428721121, 'nDCG@1000': 0.7222787036680871, 'R(rel=2)@3': 0.17539344587574904, 'R(rel=2)@5': 0.27378682716659314, 'R(rel=2)@10': 0.39822296447852207, 'R(rel=2)@25': 0.5556385033088062, 'R(rel=2)@50': 0.6714160684731278, 'R(rel=2)@100': 0.7395599257686724, 'R(rel=2)@200': 0.7984849084532355, 'R(rel=2)@1000': 0.8720241702758739, 'P(rel=2)@1': 0.7777777777777778, 'P(rel=2)@3': 0.7037037037037036, 'P(rel=2)@5': 0.6481481481481483, 'P(rel=2)@10': 0.5444444444444444, 'P(rel=2)@25': 0.38592592592592595, 'P(rel=2)@50': 0.26925925925925925, 'P(rel=2)@100': 0.16537037037037036, 'P(rel=2)@200': 0.09379629629629632, 'P(rel=2)@1000': 0.022203703703703705, 'AP(rel=2)@1000': 0.5005640745825336, 'RR(rel=2)@10': 0.8719135802469137, 'NumRet': 54000.0, 'num_q': 54.0}

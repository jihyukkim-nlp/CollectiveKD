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

   NDCG@10 = 0.430
   MAP@1000 = 0.372

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

# Newly Experimented (Fine-tuning ColBERT; bsize 18, # of HN 4)

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
[Nov 16, 14:27:32] #> NDCG@10 = 0.4394903455489325
[Nov 16, 14:27:32] #> MAP@1000 = 0.37945344982997103

TREC 2019: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn/TREC2019-psg/rerank.py/2021-10-12_14.24.55/ranking.tsv)
{'nDCG@10': 0.7414371749888692, 'nDCG@25': 0.6912861283638474, 'nDCG@50': 0.6651141257207117, 'nDCG@100': 0.6546823830897742, 'nDCG@200': 0.6682710830890339, 'nDCG@500': 0.7006364918196664, 'nDCG@1000': 0.717662830416884, 'R(rel=2)@3': 0.14006075962131528, 'R(rel=2)@5': 0.18914642611621504, 'R(rel=2)@10': 0.2894266186254735, 'R(rel=2)@25': 0.43227938979752983, 'R(rel=2)@50': 0.5325804141762943, 'R(rel=2)@100': 0.6262149523371159, 'R(rel=2)@200': 0.7176619981791489, 'R(rel=2)@1000': 0.8378453338575937, 'P(rel=2)@3': 0.7751937984496124, 'P(rel=2)@5': 0.7209302325581395, 'P(rel=2)@10': 0.6441860465116279, 'P(rel=2)@25': 0.49209302325581405, 'P(rel=2)@50': 0.36511627906976757, 'P(rel=2)@100': 0.2653488372093023, 'P(rel=2)@200': 0.17302325581395347, 'AP(rel=2)@1000': 0.48508580264653356, 'RR(rel=2)@10': 0.9031007751937985, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/rerank.py/2021-10-12_14.24.58/ranking.tsv)
{'nDCG@10': 0.6998388784820837, 'nDCG@25': 0.6701788797695977, 'nDCG@50': 0.6520576880566012, 'nDCG@100': 0.6495910621986439, 'nDCG@200': 0.6762072849837231, 'nDCG@500': 0.7025032136354434, 'nDCG@1000': 0.7159782428303274, 'R(rel=2)@3': 0.1754122377184954, 'R(rel=2)@5': 0.2592727410388973, 'R(rel=2)@10': 0.3869865168153549, 'R(rel=2)@25': 0.5604024403846639, 'R(rel=2)@50': 0.6675922731394245, 'R(rel=2)@100': 0.7405625934528436, 'R(rel=2)@200': 0.7985698819377204, 'R(rel=2)@1000': 0.8670388401182964, 'P(rel=2)@3': 0.7098765432098765, 'P(rel=2)@5': 0.6481481481481484, 'P(rel=2)@10': 0.5425925925925926, 'P(rel=2)@25': 0.38074074074074066, 'P(rel=2)@50': 0.2637037037037037, 'P(rel=2)@100': 0.16407407407407407, 'P(rel=2)@200': 0.09490740740740743, 'AP(rel=2)@1000': 0.49370822262673547, 'RR(rel=2)@10': 0.8370370370370371, 'NumRet': 54000.0, 'num_q': 54.0}

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
{'nDCG@10': 0.73821514263214, 'nDCG@25': 0.6991920318931583, 'nDCG@50': 0.6809357262843837, 'nDCG@100': 0.6642373026590526, 'nDCG@200': 0.6775029129854387, 'nDCG@500': 0.7147231239649655, 'nDCG@1000': 0.7365887269526392, 'R(rel=2)@3': 0.13995974121890126, 'R(rel=2)@5': 0.1879646468315362, 'R(rel=2)@10': 0.2863952344404925, 'R(rel=2)@25': 0.43590176189831553, 'R(rel=2)@50': 0.5532661782336662, 'R(rel=2)@100': 0.6452305747586998, 'R(rel=2)@200': 0.7288089905084988, 'R(rel=2)@1000': 0.8580758007807701, 'P(rel=2)@3': 0.7829457364341086, 'P(rel=2)@5': 0.7209302325581395, 'P(rel=2)@10': 0.6558139534883721, 'P(rel=2)@25': 0.4967441860465116, 'P(rel=2)@50': 0.3776744186046513, 'P(rel=2)@100': 0.2634883720930232, 'P(rel=2)@200': 0.1731395348837209, 'AP(rel=2)@1000': 0.4958009521019259, 'RR(rel=2)@10': 0.8953488372093024, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe/TREC2020-psg/rerank.py/2021-10-12_14.38.29/ranking.tsv)
{'nDCG@10': 0.7000146727906563, 'nDCG@25': 0.6682361690445665, 'nDCG@50': 0.6546623003029217, 'nDCG@100': 0.6535022780394425, 'nDCG@200': 0.6753943148077853, 'nDCG@500': 0.7006928350514428, 'nDCG@1000': 0.7169336453605907, 'R(rel=2)@3': 0.16750732753859002, 'R(rel=2)@5': 0.254784134804742, 'R(rel=2)@10': 0.3944522127444299, 'R(rel=2)@25': 0.5520580951501775, 'R(rel=2)@50': 0.6796635608780147, 'R(rel=2)@100': 0.7541614293992663, 'R(rel=2)@200': 0.7987953759038166, 'R(rel=2)@1000': 0.8737366235789776, 'P(rel=2)@3': 0.6604938271604939, 'P(rel=2)@5': 0.625925925925926, 'P(rel=2)@10': 0.5462962962962964, 'P(rel=2)@25': 0.38370370370370366, 'P(rel=2)@50': 0.26629629629629636, 'P(rel=2)@100': 0.1668518518518519, 'P(rel=2)@200': 0.09370370370370372, 'AP(rel=2)@1000': 0.4932469241135663, 'RR(rel=2)@10': 0.8503086419753088, 'NumRet': 54000.0, 'num_q': 54.0}

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
[Oct 28, 17:22:18] #> MAP@1000 = 0.3905157204374756

TREC 2019: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/TREC2019-psg/rerank.py/2021-10-28_17.09.53/ranking.tsv)
{'nDCG@10': 0.7444657154258438, 'nDCG@25': 0.700609112467417, 'nDCG@50': 0.6782362779967904, 'nDCG@100': 0.6625906052930649, 'nDCG@200': 0.67660462384775, 'nDCG@500': 0.7081542035478159, 'nDCG@1000': 0.7284524224158535, 'R(rel=2)@3': 0.13721996424325578, 'R(rel=2)@5': 0.19419671633821498, 'R(rel=2)@10': 0.2867178702758005, 'R(rel=2)@25': 0.43487560616983434, 'R(rel=2)@50': 0.5416819225785917, 'R(rel=2)@100': 0.6317803173067645, 'R(rel=2)@200': 0.7262171594511034, 'R(rel=2)@1000': 0.8431706927747813, 'P(rel=2)@3': 0.7674418604651163, 'P(rel=2)@5': 0.7395348837209302, 'P(rel=2)@10': 0.6558139534883721, 'P(rel=2)@25': 0.4948837209302326, 'P(rel=2)@50': 0.3767441860465117, 'P(rel=2)@100': 0.2644186046511628, 'P(rel=2)@200': 0.17395348837209298, 'AP(rel=2)@1000': 0.4917772900026287, 'RR(rel=2)@10': 0.8992248062015503, 'NumRet': 43000.0, 'num_q': 43.0}

TREC 2020: End-to-end ranking performance (experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5/TREC2020-psg/rerank.py/2021-10-28_17.09.50/ranking.tsv)
{'nDCG@10': 0.7242514449292434, 'nDCG@25': 0.6874271292815659, 'nDCG@50': 0.6689929685326551, 'nDCG@100': 0.6634512390168448, 'nDCG@200': 0.6842849149440485, 'nDCG@500': 0.7162029357361335, 'nDCG@1000': 0.73017988905081, 'R(rel=2)@3': 0.1774525556830816, 'R(rel=2)@5': 0.2952743935120175, 'R(rel=2)@10': 0.40503055955032585, 'R(rel=2)@25': 0.5728708073062028, 'R(rel=2)@50': 0.6720374350007257, 'R(rel=2)@100': 0.7425916514932751, 'R(rel=2)@200': 0.7877326007000796, 'R(rel=2)@1000': 0.8738134872402924, 'P(rel=2)@3': 0.7345679012345678, 'P(rel=2)@5': 0.7037037037037038, 'P(rel=2)@10': 0.5611111111111111, 'P(rel=2)@25': 0.3866666666666665, 'P(rel=2)@50': 0.2633333333333333, 'P(rel=2)@100': 0.16592592592592595, 'P(rel=2)@200': 0.09342592592592594, 'AP(rel=2)@1000': 0.508362088048785, 'RR(rel=2)@10': 0.8996913580246914, 'NumRet': 54000.0, 'num_q': 54.0}

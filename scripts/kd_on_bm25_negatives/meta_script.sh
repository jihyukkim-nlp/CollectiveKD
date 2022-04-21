""" Baseline: ColBERT (reported in the original paper - training using pairwise negatives)
Re-ranking
   MRR@10: 0.349
End-to-end ranking
   MRR@10 = 0.360
   Recall@50 = 0.829
   Recall@200 = 0.923
   Recall@1000 = 0.968
"""
""" Baseline: ColBERT (our implementation - trained using in-batch negatives) (colbert.teacher.dnn)
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
"""

# Precondition: collective feedbacks for queries
# experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta1.0/label.py/expansion.pt
# experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta0.5/label.py/expansion.pt

#* scp -P 7777 -r sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume
# Training with KD using labeled positives and BM25 negatives
devices=6,7 # e.g., "0,1"
master_port=29500 # e.g., "29500"
exp_root=experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt
sh scripts/kd_on_bm25_negatives/msmarco_psg.training.kd.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt}
# validation
mkdir -p experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_sonic.sh experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n 400000
sh scripts/validation/msmarco_psg.sh 4 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n 400000 > experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/400000.log
ls experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/ | grep ".log"
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/25000.log  | head -2 # MRR@10 = 0.33041018556419643
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/50000.log  | head -2 # MRR@10 = 0.34268408605084816
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/75000.log  | head -2 # MRR@10 = 0.34468993041342644
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/100000.log | head -2 # MRR@10 = 0.34806186610269657
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/350000.log | head -2 # MRR@10 = 0.35312258379951766
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/test.py/400000.log | head -2 # MRR@10 = 0.35356921135216257 #* Best
# ls experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/ # colbert-400000.dnn
# 
# Resume training
devices=6,7 # e.g., "0,1"
master_port=29500 # e.g., "29500"
exp_root=experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt
prevstep=400000
maxsteps=600000
sh scripts/kd_on_bm25_negatives/msmarco_psg.training.kd.resume.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt} ${prevstep} ${maxsteps}
# validation
mkdir -p experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_sonic.sh experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume 400000
sh scripts/validation/msmarco_psg.sh 6 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume 600000 > experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/600000.log
ls experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/ | grep ".log"
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/450000.log | head -2 # MRR@10 = 0.35198145495065275
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/500000.log | head -2 # MRR@10 = 0.35238663755855726
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/550000.log | head -2 # MRR@10 = 0.355203756765361
tail -15 experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/test.py/600000.log | head -2 # MRR@10 = 0.3555993882748899
# indexing
indexing_devices=0,1,2,3,4,5,6,7
faiss_devices=0,1
exp_root=experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume
step=600000
sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}
# retrieve & rerank & evaluation
exp_root=experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume
step=600000
device=6
sh scripts/ranking/msmarco_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/MSMARCO-psg/rerank.py/2021-11-08_05.32.34/ranking.tsv
[Jan 02, 20:25:50] #> MRR@10 = 0.3668845226724886
[Jan 02, 20:25:50] #> MRR@100 = 0.3774753033318968
[Jan 02, 20:25:50] #> Recall@50 = 0.8398758357211079
[Jan 02, 20:25:50] #> Recall@200 = 0.92804441260745
[Jan 02, 20:25:50] #> Recall@1000 = 0.9689589302769821
[Jan 02, 20:25:50] #> NDCG@10 = 0.43130531355707524
[Jan 02, 20:25:50] #> MAP@1000 = 0.3723880870785221
"""
sh scripts/ranking/trec2019_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/TREC2019-psg/rerank.py/2021-11-08_05.41.27/ranking.tsv
{'nDCG@10': 0.7141985100157044, 'nDCG@25': 0.6786274174187298, 'nDCG@50': 0.6573840479065987, 'nDCG@100': 0.6471147691957185, 'nDCG@200': 0.6551063682869122, 'nDCG@500': 0.691297936629564, 'nDCG@1000': 0.7146281457796366, 'R(rel=2)@3': 0.13271753440064185, 'R(rel=2)@5': 0.1815201244780154, 'R(rel=2)@10': 0.2910619705801395, 'R(rel=2)@25': 0.4377554951654097, 'R(rel=2)@50': 0.5410837428916153, 'R(rel=2)@100': 0.6334262661296539, 'R(rel=2)@200': 0.7103819417370096, 'R(rel=2)@1000': 0.8402365842683676, 'P(rel=2)@1': 0.7906976744186046, 'P(rel=2)@3': 0.7596899224806203, 'P(rel=2)@5': 0.7116279069767443, 'P(rel=2)@10': 0.6395348837209304, 'P(rel=2)@25': 0.4920930232558138, 'P(rel=2)@50': 0.3679069767441861, 'P(rel=2)@100': 0.2609302325581394, 'P(rel=2)@200': 0.16930232558139532, 'P(rel=2)@1000': 0.04379069767441859, 'AP(rel=2)@1000': 0.4805637541521925, 'RR(rel=2)@10': 0.882170542635659, 'NumRet': 43000.0, 'num_q': 43.0}
"""
sh scripts/ranking/trec2020_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/prf.beta0.5.b36.lr3e6.bm25n.resume/TREC2020-psg/rerank.py/2021-11-08_05.47.47/ranking.tsv
{'nDCG@10': 0.6943876754770782, 'nDCG@25': 0.6544723125435522, 'nDCG@50': 0.6358738526487643, 'nDCG@100': 0.6364062310837026, 'nDCG@200': 0.6573659948601213, 'nDCG@500': 0.6849415360728779, 'nDCG@1000': 0.698014208448687, 'R(rel=2)@3': 0.18780201648394007, 'R(rel=2)@5': 0.254599013840508, 'R(rel=2)@10': 0.3844106953671441, 'R(rel=2)@25': 0.5389420558966719, 'R(rel=2)@50': 0.6514883893072727, 'R(rel=2)@100': 0.7252249590663392, 'R(rel=2)@200': 0.784373897624689, 'R(rel=2)@1000': 0.8500364436912545, 'P(rel=2)@1': 0.7592592592592593, 'P(rel=2)@3': 0.7098765432098766, 'P(rel=2)@5': 0.6370370370370372, 'P(rel=2)@10': 0.5388888888888889, 'P(rel=2)@25': 0.3733333333333334, 'P(rel=2)@50': 0.2548148148148148, 'P(rel=2)@100': 0.1574074074074074, 'P(rel=2)@200': 0.09018518518518517, 'P(rel=2)@1000': 0.021222222222222226, 'AP(rel=2)@1000': 0.4815734262572153, 'RR(rel=2)@10': 0.8467813051146384, 'NumRet': 54000.0, 'num_q': 54.0}
"""




#* BM25 & KD from Collective Feedback Encoder
#* scp -P 7777 -r sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n
# Training with KD using labeled positives and BM25 negatives
devices=4,5
master_port=29600
exp_root=experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt
maxsteps=600000
sh scripts/kd_on_bm25_negatives/msmarco_psg.training.kd.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt} ${maxsteps}
# validation
mkdir -p experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints
mkdir -p experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py
./scripts/utils/checkpoint_scp_from_sonic.sh experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n 200000
sh scripts/validation/msmarco_psg.sh 6 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n 200000 > experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/200000.log
ls experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/ | grep ".log"
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/100000.log | head -2 # MRR@10 = 0.3452139332332741
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/200000.log | head -2 # MRR@10 = 0.3513765520534862
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/300000.log | head -2 # MRR@10 = 0.35267885568745094
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/400000.log | head -2 # MRR@10 = 0.35351770364306107
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/450000.log | head -2 # MRR@10 = 0.3537730818210763 #* Best
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/500000.log | head -2 # MRR@10 = 0.3522831673261471
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/550000.log | head -2 # MRR@10 = 0.3529164392595624
tail -15 experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/test.py/600000.log | head -2 # MRR@10 = 0.3517901487242457
ls experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/ # colbert-450000.dnn
# Remove checkpoints, except the optimal checkpoint
sh scripts/utils/remove_checkpoints.sh experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n 450000
# Indexing
indexing_devices=0,1,2,3,4,5,6,7
faiss_devices=0,1
exp_root=experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n
step=450000
sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}
# Retrieve & Rerank & Evaluation
exp_root=experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n
step=450000
device=7
sh scripts/ranking/msmarco_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/rerank.py/2022-01-02_21.40.54/ranking.tsv
[Jan 03, 01:26:51] #> MRR@10 = 0.36645148269431954
[Jan 03, 01:26:51] #> MRR@100 = 0.3768738750560674
[Jan 03, 01:26:51] #> Recall@50 = 0.838872970391595
[Jan 03, 01:26:51] #> Recall@200 = 0.9286413562559697
[Jan 03, 01:26:51] #> Recall@1000 = 0.9681351480420249
[Jan 03, 01:26:51] #> NDCG@10 = 0.4310401513819159
[Jan 03, 01:26:51] #> MAP@1000 = 0.371838597278874
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 1 --per_query_annotate --queries data/msmarco-pass/queries.dev.small.tsv \
--qrels data/msmarco-pass/qrels.dev.small.tsv --ranking experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/rerank.py/2022-01-02_21.40.54/ranking.tsv
# experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/rerank.py/2022-01-02_21.40.54/ranking.tsv.per_query.metrics
# 
sh scripts/ranking/trec2019_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2019-psg/rerank.py/2022-01-02_21.49.53/ranking.tsv
{'nDCG@10': 0.7174908822644547, 'nDCG@25': 0.6775001275692492, 'nDCG@50': 0.6616200122251681, 'nDCG@100': 0.6495032742257831, 'nDCG@200': 0.6623916947755903, 'nDCG@500': 0.6956155207504072, 'nDCG@1000': 0.7182353161677603, 'R(rel=2)@3': 0.13744066535558858, 'R(rel=2)@5': 0.18849050282909613, 'R(rel=2)@10': 0.28102819776158716, 'R(rel=2)@25': 0.43082988998762245, 'R(rel=2)@50': 0.5413332113088831, 'R(rel=2)@100': 0.6330495061057736, 'R(rel=2)@200': 0.7160486596901612, 'R(rel=2)@1000': 0.8378076838280116, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.7596899224806203, 'P(rel=2)@5': 0.7069767441860466, 'P(rel=2)@10': 0.6395348837209304, 'P(rel=2)@25': 0.4874418604651163, 'P(rel=2)@50': 0.3679069767441861, 'P(rel=2)@100': 0.2606976744186047, 'P(rel=2)@200': 0.16988372093023252, 'P(rel=2)@1000': 0.0439302325581395, 'AP(rel=2)@1000': 0.4850671339215074, 'RR(rel=2)@10': 0.8391472868217055, 'NumRet': 43000.0, 'num_q': 43.0}
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2019.tsv \
--qrels data/trec2019/2019qrels-pass.txt --ranking experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2019-psg/rerank.py/2022-01-02_21.49.53/ranking.tsv
# experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2019-psg/rerank.py/2022-01-02_21.49.53/ranking.tsv.per_query.metrics
# 
sh scripts/ranking/trec2020_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2020-psg/rerank.py/2022-01-02_21.53.48/ranking.tsv
{'nDCG@10': 0.703819256831663, 'nDCG@25': 0.6682984785852627, 'nDCG@50': 0.6468121560490849, 'nDCG@100': 0.6472374045137932, 'nDCG@200': 0.6690907112646011, 'nDCG@500': 0.6951255122136103, 'nDCG@1000': 0.7072126795152885, 'R(rel=2)@3': 0.1831447602005623, 'R(rel=2)@5': 0.2603933083427227, 'R(rel=2)@10': 0.3915171010090546, 'R(rel=2)@25': 0.5713852571661407, 'R(rel=2)@50': 0.6620813135784963, 'R(rel=2)@100': 0.7370983879298397, 'R(rel=2)@200': 0.7958410825612781, 'R(rel=2)@1000': 0.8516861422945796, 'P(rel=2)@1': 0.7222222222222222, 'P(rel=2)@3': 0.7160493827160493, 'P(rel=2)@5': 0.6518518518518519, 'P(rel=2)@10': 0.5592592592592592, 'P(rel=2)@25': 0.38592592592592584, 'P(rel=2)@50': 0.25777777777777783, 'P(rel=2)@100': 0.15981481481481477, 'P(rel=2)@200': 0.09166666666666667, 'P(rel=2)@1000': 0.021481481481481483, 'AP(rel=2)@1000': 0.4908452155600745, 'RR(rel=2)@10': 0.8277777777777778, 'NumRet': 54000.0, 'num_q': 54.0}
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2020.tsv \
--qrels data/trec2020/2020qrels-pass.txt --ranking experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2020-psg/rerank.py/2022-01-02_21.53.48/ranking.tsv
# experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2020-psg/rerank.py/2022-01-02_21.53.48/ranking.tsv.per_query.metrics
# Remove index
rm -r experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/index.py/MSMARCO.L2.32x200k





#* Training with static KD; pseudo-labels from a cross encoder or an ensemble of cross-encoders
#* static KD from an ensemble of cross-encoders
devices=4,5
master_port=29500
exp_root=experiments/kd_on_bm25_negatives/static_kd/ce_ensemble
maxsteps=600000
triples=data/msmarco-pass/triples.bm25n.sebastian_ensemble.jsonl
static_supervision=data/msmarco-pass/cross_encoder_scores/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv
sh scripts/kd_on_bm25_negatives/msmarco_psg.training.static_kd.sh ${devices} ${master_port} ${exp_root} ${maxsteps} ${triples} ${static_supervision}
# Validation
sh scripts/validation/msmarco_psg.sh 6 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble 600000
# 
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/100000.log | head -2 # MRR@10 = 0.34799711192977667
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/150000.log | head -2 # MRR@10 = 0.35252586755810245
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/200000.log | head -2 # MRR@10 = 0.3561312366398329
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/250000.log | head -2 # MRR@10 = 0.3589179969982262
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/300000.log | head -2 # MRR@10 = 0.3597101696456994
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/350000.log | head -2 # MRR@10 = 0.3629046709419199
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/400000.log | head -2 # MRR@10 = 0.36191795151680506
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/450000.log | head -2 # MRR@10 = 0.36275839132214505
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/500000.log | head -2 # MRR@10 = 0.36321280756810836
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/550000.log | head -2 # MRR@10 = 0.3668363692181744 #* Best
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/600000.log | head -2 # MRR@10 = 0.36397052803929547
# 
tail -30 experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/600000.log
cat experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/test.py/550000.log | grep -A 5 "\[*\] 1254 "
# Remove checkpoints, except the optimal checkpoint
sh scripts/utils/remove_checkpoints.sh experiments/kd_on_bm25_negatives/static_kd/ce_ensemble 550000
# Indexing
indexing_devices=4,5,6,7
faiss_devices=6,7
exp_root=experiments/kd_on_bm25_negatives/static_kd/ce_ensemble
step=550000
sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}
# Retrieve & Rerank & Evaluation
exp_root=experiments/kd_on_bm25_negatives/static_kd/ce_ensemble
step=550000
device=6
sh scripts/ranking/msmarco_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/rerank.py/2022-01-06_07.29.24/ranking.tsv
[Jan 06, 07:37:16] #> MRR@10 = 0.3781192863965068
[Jan 06, 07:37:16] #> MRR@100 = 0.38831940881356203
[Jan 06, 07:37:16] #> Recall@50 = 0.8390759312320917
[Jan 06, 07:37:16] #> Recall@200 = 0.9209169054441262
[Jan 06, 07:37:16] #> Recall@1000 = 0.9627268385864377
[Jan 06, 07:37:16] #> NDCG@10 = 0.4409390989990921
[Jan 06, 07:37:16] #> MAP@1000 = 0.38356632464928114
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 1 --per_query_annotate --queries data/msmarco-pass/queries.dev.small.tsv \
--qrels data/msmarco-pass/qrels.dev.small.tsv --ranking experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/rerank.py/2022-01-06_07.29.24/ranking.tsv
# experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/rerank.py/2022-01-06_07.29.24/ranking.tsv.per_query.metrics
# 
sh scripts/ranking/trec2019_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2019-psg/rerank.py/2022-01-06_06.29.16/ranking.tsv
{'nDCG@10': 0.7168998764033627, 'nDCG@25': 0.6615780287003624, 'nDCG@50': 0.627980891981794, 'nDCG@100': 0.611223132211963, 'nDCG@200': 0.6230093275454529, 'nDCG@500': 0.6584988416838347, 'nDCG@1000': 0.6777694785006012, 'R(rel=2)@3': 0.1336381512219627, 'R(rel=2)@5': 0.18641464117134904, 'R(rel=2)@10': 0.291710681745938, 'R(rel=2)@25': 0.4173442364604556, 'R(rel=2)@50': 0.5204692610925133, 'R(rel=2)@100': 0.5982070189164587, 'R(rel=2)@200': 0.67508936957579, 'R(rel=2)@1000': 0.7932729875209703, 'P(rel=2)@1': 0.7674418604651163, 'P(rel=2)@3': 0.7751937984496127, 'P(rel=2)@5': 0.7302325581395351, 'P(rel=2)@10': 0.6488372093023257, 'P(rel=2)@25': 0.47441860465116276, 'P(rel=2)@50': 0.34837209302325584, 'P(rel=2)@100': 0.24139534883720928, 'P(rel=2)@200': 0.15720930232558136, 'P(rel=2)@1000': 0.04025581395348835, 'AP(rel=2)@1000': 0.46272761427142883, 'RR(rel=2)@10': 0.8651162790697675, 'NumRet': 43000.0, 'num_q': 43.0}
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2019.tsv \
--qrels data/trec2019/2019qrels-pass.txt --ranking experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2019-psg/rerank.py/2022-01-06_06.29.16/ranking.tsv
# experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2019-psg/rerank.py/2022-01-06_06.29.16/ranking.tsv.per_query.metrics
# 
sh scripts/ranking/trec2020_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2020-psg/rerank.py/2022-01-06_06.33.35/ranking.tsv
{'nDCG@10': 0.6842650994433386, 'nDCG@25': 0.6377102444142558, 'nDCG@50': 0.6192171532462022, 'nDCG@100': 0.6123737065251846, 'nDCG@200': 0.6305238336331008, 'nDCG@500': 0.6584319160478364, 'nDCG@1000': 0.6718384276490689, 'R(rel=2)@3': 0.17928724495882217, 'R(rel=2)@5': 0.24999452614421902, 'R(rel=2)@10': 0.3751647455619935, 'R(rel=2)@25': 0.5235235150517856, 'R(rel=2)@50': 0.6271526986055427, 'R(rel=2)@100': 0.7091574616665921, 'R(rel=2)@200': 0.7506244924167761, 'R(rel=2)@1000': 0.823582978226531, 'P(rel=2)@1': 0.7777777777777778, 'P(rel=2)@3': 0.691358024691358, 'P(rel=2)@5': 0.614814814814815, 'P(rel=2)@10': 0.5240740740740741, 'P(rel=2)@25': 0.3659259259259259, 'P(rel=2)@50': 0.2537037037037036, 'P(rel=2)@100': 0.1546296296296296, 'P(rel=2)@200': 0.08638888888888888, 'P(rel=2)@1000': 0.02064814814814815, 'AP(rel=2)@1000': 0.4601593738240795, 'RR(rel=2)@10': 0.8510288065843622, 'NumRet': 54000.0, 'num_q': 54.0}
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2020.tsv \
--qrels data/trec2020/2020qrels-pass.txt --ranking experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2020-psg/rerank.py/2022-01-06_06.33.35/ranking.tsv
# experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2020-psg/rerank.py/2022-01-06_06.33.35/ranking.tsv.per_query.metrics
# 
# Remove index
rm -r experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/index.py/MSMARCO.L2.32x200k




#* static KD from a single cross-encoder
devices=6,7
master_port=29600
exp_root=experiments/kd_on_bm25_negatives/static_kd/ce_single
maxsteps=600000
triples=data/msmarco-pass/triples.bm25n.sebastian_single.jsonl
static_supervision=data/msmarco-pass/cross_encoder_scores/bertbase_cat_msmarcopassage_train_scores_ids.tsv
sh scripts/kd_on_bm25_negatives/msmarco_psg.training.static_kd.sh ${devices} ${master_port} ${exp_root} ${maxsteps} ${triples} ${static_supervision}
# validation
sh scripts/validation/msmarco_psg.sh 7 experiments/kd_on_bm25_negatives/static_kd/ce_single 600000
# 
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/100000.log | head -2 # MRR@10 = 0.3459963273752665
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/150000.log | head -2 # MRR@10 = 0.3519885045708829
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/200000.log | head -2 # MRR@10 = 0.35225599217719583
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/250000.log | head -2 # MRR@10 = 0.3555383863191877
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/300000.log | head -2 # MRR@10 = 0.35637035520989685
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/350000.log | head -2 # MRR@10 = 0.3580358848410424
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/400000.log | head -2 # MRR@10 = 0.35481113839996403
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/450000.log | head -2 # MRR@10 = 0.3581000704962023
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/500000.log | head -2 # MRR@10 = 0.36078120025469534
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/550000.log | head -2 # MRR@10 = 0.3594895279028514
tail -15 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/600000.log | head -2 # MRR@10 = 0.36111838813844477 #* Best
# 
tail -30 experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/600000.log
cat experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/test.py/500000.log | grep -A 5 "\[*\] 1215 "
# Remove checkpoints, except the optimal checkpoint
sh scripts/utils/remove_checkpoints.sh experiments/kd_on_bm25_negatives/static_kd/ce_single 600000
# Indexing
indexing_devices=4,5,6,7
faiss_devices=6,7
exp_root=experiments/kd_on_bm25_negatives/static_kd/ce_single
step=600000
sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}
# Retrieve & Rerank & Evaluation
exp_root=experiments/kd_on_bm25_negatives/static_kd/ce_single
step=600000
device=7
sh scripts/ranking/msmarco_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/rerank.py/2022-01-06_09.09.09/ranking.tsv
[Jan 06, 09:53:33] #> MRR@10 = 0.37291518852049016
[Jan 06, 09:53:33] #> MRR@100 = 0.38364696006707644
[Jan 06, 09:53:33] #> Recall@50 = 0.8376313276026742
[Jan 06, 09:53:33] #> Recall@200 = 0.9262893982808024
[Jan 06, 09:53:33] #> Recall@1000 = 0.9666786055396372
[Jan 06, 09:53:33] #> NDCG@10 = 0.4347125226419643
[Jan 06, 09:53:33] #> MAP@1000 = 0.3782918890958763
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 1 --per_query_annotate --queries data/msmarco-pass/queries.dev.small.tsv \
--qrels data/msmarco-pass/qrels.dev.small.tsv --ranking experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/rerank.py/2022-01-06_09.09.09/ranking.tsv
# experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/rerank.py/2022-01-06_09.09.09/ranking.tsv.per_query.metrics
# 
sh scripts/ranking/trec2019_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2019-psg/rerank.py/2022-01-06_09.21.09/ranking.tsv
{'nDCG@10': 0.7145625592593483, 'nDCG@25': 0.6645724292900926, 'nDCG@50': 0.6323908619792923, 'nDCG@100': 0.6212917685465073, 'nDCG@200': 0.6316978352444873, 'nDCG@500': 0.6644083525921656, 'nDCG@1000': 0.68081782161689, 'R(rel=2)@3': 0.13724539306547978, 'R(rel=2)@5': 0.19075432290294606, 'R(rel=2)@10': 0.27578380002657243, 'R(rel=2)@25': 0.42860674306167956, 'R(rel=2)@50': 0.5138209260384121, 'R(rel=2)@100': 0.6041425742723469, 'R(rel=2)@200': 0.6862814629890004, 'R(rel=2)@1000': 0.7906020102753923, 'P(rel=2)@1': 0.7906976744186046, 'P(rel=2)@3': 0.7829457364341086, 'P(rel=2)@5': 0.7255813953488374, 'P(rel=2)@10': 0.6209302325581395, 'P(rel=2)@25': 0.4753488372093024, 'P(rel=2)@50': 0.34279069767441867, 'P(rel=2)@100': 0.24441860465116277, 'P(rel=2)@200': 0.15802325581395343, 'P(rel=2)@1000': 0.03979069767441857, 'AP(rel=2)@1000': 0.45606256228953534, 'RR(rel=2)@10': 0.8798449612403101, 'NumRet': 43000.0, 'num_q': 43.0}
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2019.tsv \
--qrels data/trec2019/2019qrels-pass.txt --ranking experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2019-psg/rerank.py/2022-01-06_09.21.09/ranking.tsv
# experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2019-psg/rerank.py/2022-01-06_09.21.09/ranking.tsv.per_query.metrics
# 
sh scripts/ranking/trec2020_psg.ranking.sh ${exp_root} ${step} ${device}
""" experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2020-psg/rerank.py/2022-01-06_09.25.42/ranking.tsv 
{'nDCG@10': 0.6955211269713317, 'nDCG@25': 0.6558970191188555, 'nDCG@50': 0.633509373107544, 'nDCG@100': 0.6311241602719327, 'nDCG@200': 0.6496289873455863, 'nDCG@500': 0.6758614289612679, 'nDCG@1000': 0.6893763509926393, 'R(rel=2)@3': 0.18453533003463274, 'R(rel=2)@5': 0.24793224517739318, 'R(rel=2)@10': 0.3725518745410624, 'R(rel=2)@25': 0.5389123950614935, 'R(rel=2)@50': 0.632822893572373, 'R(rel=2)@100': 0.7237206160391239, 'R(rel=2)@200': 0.7660193784215755, 'R(rel=2)@1000': 0.8289498042439868, 'P(rel=2)@1': 0.7592592592592593, 'P(rel=2)@3': 0.7160493827160493, 'P(rel=2)@5': 0.6333333333333334, 'P(rel=2)@10': 0.5203703703703704, 'P(rel=2)@25': 0.3674074074074073, 'P(rel=2)@50': 0.25333333333333335, 'P(rel=2)@100': 0.15777777777777777, 'P(rel=2)@200': 0.08833333333333332, 'P(rel=2)@1000': 0.020851851851851858, 'AP(rel=2)@1000': 0.4635232593221489, 'RR(rel=2)@10': 0.8561728395061728, 'NumRet': 54000.0, 'num_q': 54.0}
"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2020.tsv \
--qrels data/trec2020/2020qrels-pass.txt --ranking experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2020-psg/rerank.py/2022-01-06_09.25.42/ranking.tsv 
# experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2020-psg/rerank.py/2022-01-06_09.25.42/ranking.tsv.per_query.metrics
# 
# Remove index
rm -r experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/index.py/MSMARCO.L2.32x200k






# Training with KD using labeled positives and BM25 negatives with dual supervision
devices=0,1
master_port=29500
exp_root=experiments/kd_on_bm25_negatives/dual_supervision/lambda0.5.ce_single.prf.beta1.0.b36.lr3e6.bm25n
kd_expansion_pt=experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta1.0/label.py/expansion.pt
maxsteps=600000
triples=data/msmarco-pass/triples.bm25n.sebastian_ensemble.jsonl
static_supervision=data/msmarco-pass/cross_encoder_scores/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv
dual_supervision_lambda=0.75
sh scripts/kd_on_bm25_negatives/msmarco_psg.training.dual_supervision.sh ${devices} ${master_port} ${exp_root} ${kd_expansion_pt} ${maxsteps} ${triples} ${static_supervision} ${dual_supervision_lambda}

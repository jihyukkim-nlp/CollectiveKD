#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg
# 
rf_fb_k=$6 #TODO: input arg
rf_fb_clusters=$7 #TODO: input arg
rf_beta=$8 #TODO: input arg

# 1. ANN Search
queries=data/queries.trec2019.tsv #TODO: custom path
index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/pilot_test/trec2019 #TODO: custom path
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return
ann_experiment=wo_qe #TODO: custom path
topk=${experiment_root}/${ann_experiment}/retrieve.py/$(ls ${experiment_root}/${ann_experiment}/retrieve.py)/unordered.tsv
if [ ! -f ${topk} ];then
    echo "1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small."
    # 1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small.
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --nprobe 32 --partitions 32768 --faiss_depth 512 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${experiment_root} --experiment ${ann_experiment}
else
    echo "We have ANN search result at: \"${topk}\""
fi



# 2. Exact-NN Search, using query expansion.
topk=${experiment_root}/${ann_experiment}/retrieve.py/$(ls ${experiment_root}/${ann_experiment}/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
# 
# ``rf_for_prf``: RF is used only for feedback documents of PRF, thus we do not use ``expansion.pt`` from RF
rerank_experiment=kmeans.rf_for_prf.rf-k${rf_fb_k}.beta${rf_beta}.clusters${rf_fb_clusters}.prf-docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
# 
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
# 
rf_exp=experiments/pilot_test/trec2019/kmeans.k${rf_fb_k}.beta${rf_beta}.clusters${rf_fb_clusters}
[ ! -d "${rf_exp}" ] && echo "${rf_exp} does not exist." && return
rf_dir=${rf_exp}/label.py/$(ls ${rf_exp}/label.py/)
fb_ranking=${rf_dir}/ranking.jsonl
[ ! -f "${fb_ranking}" ] && echo "${fb_ranking} does not exist." && return
# expansion_pt=${rf_dir}/expansion.pt
# 
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --checkpoint ${checkpoint} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    \
    --prf \
    --fb_ranking ${fb_ranking} \
    --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta}  --fb_clusters ${fb_clusters} \
    --depth 1000
else
    echo "We have Exact-NN search result at: \"${ranking_jsonl}\""
fi



# 3. Evaluation
# 3-1. Overall
ranking=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
# 
result_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics
qrels=data/pilot_test/label/2019qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
# RF (fb_k: 10, beta: 0.5, fb_clusters: 10) then PRF (fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24)
#* {'nDCG@10': 0.7900236250822773, 'nDCG@25': 0.7631403804229614, 'nDCG@50': 0.7389012703257533, 'nDCG@100': 0.7209788651724895, 'nDCG@200': 0.718721457868721, 'nDCG@500': 0.7375759756646981, 'nDCG@1000': 0.7470364893876701, 'R(rel=2)@3': 0.1677773553475498, 'R(rel=2)@5': 0.21676913353902474, 'R(rel=2)@10': 0.31885961001855595, 'R(rel=2)@25': 0.4920102286088171, 'R(rel=2)@50': 0.609290947208698, 'R(rel=2)@100': 0.6990480362583636, 'R(rel=2)@200': 0.7661202352937455, 'R(rel=2)@1000': 0.8238883621011537, 'P(rel=2)@3': 0.8294573643410852, 'P(rel=2)@5': 0.7534883720930234, 'P(rel=2)@10': 0.6790697674418604, 'P(rel=2)@25': 0.5367441860465114, 'P(rel=2)@50': 0.40558139534883725, 'P(rel=2)@100': 0.29162790697674423, 'P(rel=2)@200': 0.1813953488372093, 'AP(rel=2)@1000': 0.5449886868612445, 'RR(rel=2)@10': 0.9331395348837209, 'NumRet': 42962.0, 'num_q': 43.0}
# RF (fb_k: 5, beta: 0.5, fb_clusters: 10) then PRF (fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24)
#* {'nDCG@10': 0.772580587710234, 'nDCG@25': 0.7458633244170145, 'nDCG@50': 0.7306308038646184, 'nDCG@100': 0.7152711800955038, 'nDCG@200': 0.7179340469106174, 'nDCG@500': 0.736980745512181, 'nDCG@1000': 0.7437259634504922, 'R(rel=2)@3': 0.16622368303116566, 'R(rel=2)@5': 0.2199811623230232, 'R(rel=2)@10': 0.30693790823678324, 'R(rel=2)@25': 0.48472908115280977, 'R(rel=2)@50': 0.6065995244923694, 'R(rel=2)@100': 0.6970490452189405, 'R(rel=2)@200': 0.7738993185039926, 'R(rel=2)@1000': 0.8241948453806414, 'P(rel=2)@3': 0.8062015503875969, 'P(rel=2)@5': 0.7581395348837211, 'P(rel=2)@10': 0.658139534883721, 'P(rel=2)@25': 0.5181395348837208, 'P(rel=2)@50': 0.4046511627906976, 'P(rel=2)@100': 0.29093023255813955, 'P(rel=2)@200': 0.18511627906976746, 'AP(rel=2)@1000': 0.5419318719125322, 'RR(rel=2)@10': 0.9098837209302325, 'NumRet': 42962.0, 'num_q': 43.0}
# RF (fb_k: 10, beta: 0.5, fb_clusters: 10) for PRF (fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24)
#* {'nDCG@10': 0.7924202365377419, 'nDCG@25': 0.7544838272390387, 'nDCG@50': 0.7328321879628117, 'nDCG@100': 0.7184740105729167, 'nDCG@200': 0.7164639509299402, 'nDCG@500': 0.7371908198323026, 'nDCG@1000': 0.7470387804837736, 'R(rel=2)@3': 0.168347984115853, 'R(rel=2)@5': 0.21171510124253573, 'R(rel=2)@10': 0.31586831250516256, 'R(rel=2)@25': 0.47097871537615754, 'R(rel=2)@50': 0.5989493478141312, 'R(rel=2)@100': 0.6933952245444766, 'R(rel=2)@200': 0.7630719554842462, 'R(rel=2)@1000': 0.8241916857019869, 'P(rel=2)@3': 0.8294573643410854, 'P(rel=2)@5': 0.7395348837209303, 'P(rel=2)@10': 0.6790697674418603, 'P(rel=2)@25': 0.523720930232558, 'P(rel=2)@50': 0.39906976744186046, 'P(rel=2)@100': 0.2895348837209303, 'P(rel=2)@200': 0.18046511627906975, 'AP(rel=2)@1000': 0.5446281726651897, 'RR(rel=2)@10': 0.9490586932447398, 'NumRet': 42962.0, 'num_q': 43.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# RF (fb_k: 10, beta: 0.5, fb_clusters: 10) then PRF (fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24)
#* {'nDCG@10': 0.8244753126325591, 'nDCG@25': 0.7843416089867287, 'nDCG@50': 0.7566950094876628, 'nDCG@100': 0.7373548747194245, 'nDCG@200': 0.7340323522081047, 'nDCG@500': 0.7526339405918907, 'nDCG@1000': 0.7619002075244491, 'R(rel=2)@3': 0.16979102950254868, 'R(rel=2)@5': 0.2252481151961091, 'R(rel=2)@10': 0.3295260307034517, 'R(rel=2)@25': 0.4974388376088103, 'R(rel=2)@50': 0.6156796426668021, 'R(rel=2)@100': 0.7044126665128654, 'R(rel=2)@200': 0.769879311474801, 'R(rel=2)@1000': 0.8274711684262395, 'P(rel=2)@3': 0.8992248062015502, 'P(rel=2)@5': 0.8186046511627909, 'P(rel=2)@10': 0.727906976744186, 'P(rel=2)@25': 0.5590697674418603, 'P(rel=2)@50': 0.4190697674418605, 'P(rel=2)@100': 0.29930232558139536, 'P(rel=2)@200': 0.18534883720930237, 'AP(rel=2)@1000': 0.5724874155414553, 'RR(rel=2)@10': 0.9689922480620154, 'NumRet': 43000.0, 'num_q': 43.0}
# RF (fb_k: 5, beta: 0.5, fb_clusters: 10) then PRF (fb_docs: 3, fb_k: 5, beta: 0.5, fb_clusters: 24)
#* {'nDCG@10': 0.8021362424574208, 'nDCG@25': 0.7645314620815197, 'nDCG@50': 0.7487285702861133, 'nDCG@100': 0.7314454536904041, 'nDCG@200': 0.7329639967104614, 'nDCG@500': 0.7515697448371136, 'nDCG@1000': 0.7581870658228045, 'R(rel=2)@3': 0.16884541494764943, 'R(rel=2)@5': 0.22795966493228262, 'R(rel=2)@10': 0.31365023560485683, 'R(rel=2)@25': 0.4801688625631783, 'R(rel=2)@50': 0.611170250016413, 'R(rel=2)@100': 0.7018841282272079, 'R(rel=2)@200': 0.7781032797855846, 'R(rel=2)@1000': 0.8277929126704549, 'P(rel=2)@3': 0.8837209302325579, 'P(rel=2)@5': 0.8186046511627909, 'P(rel=2)@10': 0.6976744186046513, 'P(rel=2)@25': 0.5376744186046511, 'P(rel=2)@50': 0.41813953488372096, 'P(rel=2)@100': 0.29860465116279067, 'P(rel=2)@200': 0.1891860465116279, 'AP(rel=2)@1000': 0.5683965877679626, 'RR(rel=2)@10': 0.945736434108527, 'NumRet': 43000.0, 'num_q': 43.0}
# RF (fb_k: 10, beta: 0.5, fb_clusters: 10), PRF (fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24)
#* {'nDCG@10': 0.8106358163108603, 'nDCG@25': 0.7687644905342093, 'nDCG@50': 0.7464393832414559, 'nDCG@100': 0.7298355512124202, 'nDCG@200': 0.7271549222315097, 'nDCG@500': 0.7473572050104904, 'nDCG@1000': 0.757021650649245, 'R(rel=2)@3': 0.17181502855337794, 'R(rel=2)@5': 0.21745998819236936, 'R(rel=2)@10': 0.32175615852458145, 'R(rel=2)@25': 0.479766591544513, 'R(rel=2)@50': 0.6067859022887455, 'R(rel=2)@100': 0.6991404659712703, 'R(rel=2)@200': 0.7675937091694481, 'R(rel=2)@1000': 0.827768449059216, 'P(rel=2)@3': 0.8914728682170542, 'P(rel=2)@5': 0.7906976744186048, 'P(rel=2)@10': 0.7162790697674418, 'P(rel=2)@25': 0.5479069767441862, 'P(rel=2)@50': 0.4134883720930233, 'P(rel=2)@100': 0.2974418604651163, 'P(rel=2)@200': 0.18453488372093022, 'AP(rel=2)@1000': 0.567216846662744, 'RR(rel=2)@10': 0.9651162790697675, 'NumRet': 43000.0, 'num_q': 43.0}



#!/bin/bash
exp_root=$1 #TODO: input arg, e.g., "experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf"
step=$2 #TODO: input arg, e.g., "150000"
device=$3 #TODO: input arg, e.g., "0"
fb_docs=$4 #TODO: input arg, e.g., "3"
fb_k=$5 #TODO: input arg, e.g., "10"
beta=$6 #TODO: input arg, e.g., "1.0"
fb_clusters=$7 #TODO: input arg, e.g., "24"
fb_ranking=$8 #TODO: input arg, e.g., "data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2019.pass.jsonl"
rerank_exp_tag=$9 #TODO: input arg, e.g., "prf-dTq" (ranking results will be saved at "${exp_root}/TREC2019-psg-${rerank_exp_tag}/"")

# sanity check
[ ! -d "${exp_root}" ] && echo "${exp_root} does not exist." && return
checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
index_root=${exp_root}/MSMARCO-psg/index.py
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return
queries=data/queries.trec2019.tsv
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
qrels=data/trec2019/2019qrels-pass.txt
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return


# ANN search
topk=${exp_root}/TREC2019-psg/retrieve.py/$(ls ${exp_root}/TREC2019-psg/retrieve.py)/unordered.tsv
if [ ! -f ${topk} ];then
    echo "ANN Search using retrieve.py"
    echo "checkpoint: ${checkpoint}"
    echo "index_root: ${index_root}"
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${exp_root} --experiment TREC2019-psg
else
    echo "We have ANN search result at: \"${topk}\""
fi
echo;echo;echo



echo;echo;echo
# 0. Preprocessing: prepare tokenids for ColBERT-PRF
# check if 
# index_path=${index_root}/MSMARCO.L2.32x200k
prev_index_path=experiments/colbert.teacher/MSMARCO-psg/index.py/MSMARCO.L2.32x200k
new_index_path=${index_root}/MSMARCO.L2.32x200k
[ ! -d ${prev_index_path} ] && echo "${prev_index_path} does not exist" && return
if [ ! -f ${prev_index_path}/tokenids.docfreq ]; then
    echo "0. Preprocessing: prepare tokenids for ColBERT-PRF"
    python -m preprocessing.pseudo_labeling.prepare_tokenids_for_colbert_prf --collection ${collection} \
    --mask-punctuation --doc_maxlen 180 --index_path ${prev_index_path}
    echo;echo;echo
fi
if [ ! -f ${new_index_path}/tokenids.docfreq ]; then
    echo "Copy tokenids from the previous index (tokenids are same for all retrievers):"
    cp --verbose ${prev_index_path}/tokenids.docfreq ${new_index_path}/
    cp --verbose ${prev_index_path}/*.tokenids ${new_index_path}/
    echo;echo;echo
fi



# 2. Exact-NN Search, using query expansion.
topk=${exp_root}/TREC2019-psg/retrieve.py/$(ls ${exp_root}/TREC2019-psg/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
# 
rerank_exp_root=${exp_root}/TREC2019-psg-${rerank_exp_tag}
rerank_exp=docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
# 
ranking_jsonl=${rerank_exp_root}/${rerank_exp}/label.py/$(ls ${rerank_exp_root}/${rerank_exp}/label.py)/ranking.jsonl
# 
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --checkpoint ${checkpoint} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --root ${rerank_exp_root} --experiment ${rerank_exp} \
    --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    \
    --prf \
    --fb_ranking ${fb_ranking} \
    --fb_docs ${fb_docs} --fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
    --depth 1000
else
    echo "We have Exact-NN search result at: \"${ranking_jsonl}\""
fi



# 3. Evaluation
# 3-1. Overall
ranking=${rerank_exp_root}/${rerank_exp}/label.py/$(ls ${rerank_exp_root}/${rerank_exp}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
# 
result=${rerank_exp_root}/${rerank_exp}/label.py/$(ls ${rerank_exp_root}/${rerank_exp}/label.py)/ranking.metrics
qrels=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --ranking ${ranking} > ${result}
echo;cat ${result} | tail -1;echo



# ColBERT-PRF from docT5query feedbacks
# 
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
# 
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
# {'nDCG@10': 0.7170664065772989, 'nDCG@25': 0.6885399237312472, 'nDCG@50': 0.6772515707818462, 'nDCG@100': 0.6639466224428454, 'nDCG@200': 0.6797032084562783, 'nDCG@500': 0.7223599787469717, 'nDCG@1000': 0.7441673499417215, 'R(rel=2)@3': 0.11970301973959556, 'R(rel=2)@5': 0.17645668466218875, 'R(rel=2)@10': 0.2913883235833616, 'R(rel=2)@25': 0.4552665498427733, 'R(rel=2)@50': 0.5647626854030791, 'R(rel=2)@100': 0.6628586250724343, 'R(rel=2)@200': 0.7523146253839452, 'R(rel=2)@1000': 0.8823437948045268, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.7364341085271318, 'P(rel=2)@5': 0.7255813953488373, 'P(rel=2)@10': 0.6581395348837209, 'P(rel=2)@25': 0.5125581395348838, 'P(rel=2)@50': 0.3902325581395349, 'P(rel=2)@100': 0.27813953488372084, 'P(rel=2)@200': 0.18418604651162787, 'P(rel=2)@1000': 0.04788372093023253, 'AP(rel=2)@1000': 0.49640175429354183, 'RR(rel=2)@10': 0.8320413436692506, 'NumRet': 43000.0, 'num_q': 43.0}



# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7499298775540899, 'nDCG@25': 0.7161047380317864, 'nDCG@50': 0.6961947902802632, 'nDCG@100': 0.6845909667460557, 'nDCG@200': 0.6944938290808806, 'nDCG@500': 0.7179404322752289, 'nDCG@1000': 0.727107434675859, 'R(rel=2)@3': 0.13058248179865975, 'R(rel=2)@5': 0.18731062030468773, 'R(rel=2)@10': 0.30728196885289805, 'R(rel=2)@25': 0.46500836962934394, 'R(rel=2)@50': 0.572616248880696, 'R(rel=2)@100': 0.6656937671398719, 'R(rel=2)@200': 0.7514044508477238, 'R(rel=2)@1000': 0.8287160609184753, 'P(rel=2)@1': 0.7906976744186046, 'P(rel=2)@3': 0.7829457364341086, 'P(rel=2)@5': 0.7395348837209301, 'P(rel=2)@10': 0.6767441860465115, 'P(rel=2)@25': 0.5293023255813954, 'P(rel=2)@50': 0.3967441860465117, 'P(rel=2)@100': 0.28395348837209294, 'P(rel=2)@200': 0.18720930232558133, 'P(rel=2)@1000': 0.04302325581395347, 'AP(rel=2)@1000': 0.5193512192963524, 'RR(rel=2)@10': 0.8740310077519381, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.7207118248605747, 'nDCG@25': 0.6939030056965461, 'nDCG@50': 0.6796624486783117, 'nDCG@100': 0.6692503183120045, 'nDCG@200': 0.680003985080787, 'nDCG@500': 0.7075620466677686, 'nDCG@1000': 0.715841964898986, 'R(rel=2)@3': 0.1286832569924582, 'R(rel=2)@5': 0.17182392856999446, 'R(rel=2)@10': 0.2873143792475185, 'R(rel=2)@25': 0.44922236344342026, 'R(rel=2)@50': 0.5661801728281298, 'R(rel=2)@100': 0.6587299250250933, 'R(rel=2)@200': 0.74388167302101, 'R(rel=2)@1000': 0.8267392191882272, 'P(rel=2)@1': 0.7674418604651163, 'P(rel=2)@3': 0.7674418604651164, 'P(rel=2)@5': 0.6930232558139533, 'P(rel=2)@10': 0.6465116279069766, 'P(rel=2)@25': 0.5172093023255815, 'P(rel=2)@50': 0.393953488372093, 'P(rel=2)@100': 0.28093023255813954, 'P(rel=2)@200': 0.1858139534883721, 'P(rel=2)@1000': 0.042976744186046495, 'AP(rel=2)@1000': 0.49769181141113095, 'RR(rel=2)@10': 0.8582041343669251, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 1, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7248732641531628, 'nDCG@25': 0.702654617610427, 'nDCG@50': 0.6814587739188417, 'nDCG@100': 0.6688069700500907, 'nDCG@200': 0.6815354370546601, 'nDCG@500': 0.7058989581055485, 'nDCG@1000': 0.7149544256000085, 'R(rel=2)@3': 0.12295554297946679, 'R(rel=2)@5': 0.1818977649052595, 'R(rel=2)@10': 0.28825157599580964, 'R(rel=2)@25': 0.4501496632979157, 'R(rel=2)@50': 0.56256878731012, 'R(rel=2)@100': 0.6650160502167393, 'R(rel=2)@200': 0.7586835608717873, 'R(rel=2)@1000': 0.8271267388426072, 'P(rel=2)@1': 0.813953488372093, 'P(rel=2)@3': 0.7441860465116279, 'P(rel=2)@5': 0.7069767441860466, 'P(rel=2)@10': 0.6581395348837212, 'P(rel=2)@25': 0.5227906976744187, 'P(rel=2)@50': 0.39069767441860465, 'P(rel=2)@100': 0.2802325581395349, 'P(rel=2)@200': 0.18406976744186043, 'P(rel=2)@1000': 0.04283720930232557, 'AP(rel=2)@1000': 0.49098426771982806, 'RR(rel=2)@10': 0.8856589147286822, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 1, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.6942515642226628, 'nDCG@25': 0.6797339640291866, 'nDCG@50': 0.6646410647569112, 'nDCG@100': 0.6537534243829861, 'nDCG@200': 0.6671395227898396, 'nDCG@500': 0.6933845075095495, 'nDCG@1000': 0.7027978767097657, 'R(rel=2)@3': 0.11500112169093121, 'R(rel=2)@5': 0.17387445182142286, 'R(rel=2)@10': 0.269213210231821, 'R(rel=2)@25': 0.4439599326702025, 'R(rel=2)@50': 0.5625150788058946, 'R(rel=2)@100': 0.6651449762924323, 'R(rel=2)@200': 0.7532780229664976, 'R(rel=2)@1000': 0.8274992258823163, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.7286821705426357, 'P(rel=2)@5': 0.702325581395349, 'P(rel=2)@10': 0.6348837209302326, 'P(rel=2)@25': 0.5153488372093024, 'P(rel=2)@50': 0.39348837209302323, 'P(rel=2)@100': 0.28116279069767447, 'P(rel=2)@200': 0.1847674418604651, 'P(rel=2)@1000': 0.04283720930232557, 'AP(rel=2)@1000': 0.4781059670515211, 'RR(rel=2)@10': 0.829263565891473, 'NumRet': 43000.0, 'num_q': 43.0}



# ColBERT
#* {'nDCG@10': 0.6996191676961366, 'nDCG@25': 0.660308544811616, 'nDCG@50': 0.6468568119807828, 'nDCG@100': 0.6419724695294958, 'nDCG@200': 0.6514069510353638, 'nDCG@500': 0.6818872751153267, 'nDCG@1000': 0.6963564628761999, 'R(rel=2)@3': 0.13427849571086078, 'R(rel=2)@5': 0.1841068344211011, 'R(rel=2)@10': 0.2875825212503319, 'R(rel=2)@25': 0.4184246086356199, 'R(rel=2)@50': 0.5317990361876885, 'R(rel=2)@100': 0.6326877005713728, 'R(rel=2)@200': 0.7134365393099832, 'R(rel=2)@1000': 0.8113306150029239, 'P(rel=2)@1': 0.7209302325581395, 'P(rel=2)@3': 0.7286821705426357, 'P(rel=2)@5': 0.7069767441860467, 'P(rel=2)@10': 0.6325581395348838, 'P(rel=2)@25': 0.46604651162790695, 'P(rel=2)@50': 0.3548837209302326, 'P(rel=2)@100': 0.2574418604651163, 'P(rel=2)@200': 0.1672093023255814, 'AP(rel=2)@1000': 0.469969658251788, 'RR(rel=2)@10': 0.8330103359173127, 'NumRet': 43000.0, 'num_q': 43.0}
# 
# ColBERT-PRF
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7190334699112224, 'nDCG@25': 0.6839502718725956, 'nDCG@50': 0.6738609908412226, 'nDCG@100': 0.6639236371636662, 'nDCG@200': 0.6760336566659717, 'nDCG@500': 0.7002810879123601, 'nDCG@1000': 0.7132432985470754, 'R(rel=2)@3': 0.1316946066362192, 'R(rel=2)@5': 0.18579675923957195, 'R(rel=2)@10': 0.2963542575698357, 'R(rel=2)@25': 0.43963669315425813, 'R(rel=2)@50': 0.5634593635913152, 'R(rel=2)@100': 0.6511298978214967, 'R(rel=2)@200': 0.7445927505839498, 'R(rel=2)@1000': 0.825054253390824, 'P(rel=2)@1': 0.813953488372093, 'P(rel=2)@3': 0.7364341085271319, 'P(rel=2)@5': 0.6930232558139536, 'P(rel=2)@10': 0.6488372093023256, 'P(rel=2)@25': 0.4883720930232556, 'P(rel=2)@50': 0.37534883720930234, 'P(rel=2)@100': 0.26953488372093015, 'P(rel=2)@200': 0.18, 'AP(rel=2)@1000': 0.5057612857667395, 'RR(rel=2)@10': 0.8556201550387598, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.7047486542256665, 'nDCG@25': 0.6760746662192733, 'nDCG@50': 0.6646475601498076, 'nDCG@100': 0.6558469500080151, 'nDCG@200': 0.664943855821488, 'nDCG@500': 0.6931277853018166, 'nDCG@1000': 0.7074749098822583, 'R(rel=2)@3': 0.12917895368451973, 'R(rel=2)@5': 0.186765751487634, 'R(rel=2)@10': 0.281323294409313, 'R(rel=2)@25': 0.4331374851046292, 'R(rel=2)@50': 0.5573479560969042, 'R(rel=2)@100': 0.6430341047795333, 'R(rel=2)@200': 0.7307657180927005, 'R(rel=2)@1000': 0.8260142709059335, 'P(rel=2)@3': 0.7209302325581396, 'P(rel=2)@5': 0.6976744186046513, 'P(rel=2)@10': 0.6302325581395348, 'P(rel=2)@25': 0.4837209302325581, 'P(rel=2)@50': 0.3697674418604652, 'P(rel=2)@100': 0.26744186046511625, 'P(rel=2)@200': 0.1768604651162791, 'AP(rel=2)@1000': 0.49678467840601537, 'RR(rel=2)@10': 0.8533591731266151, 'NumRet': 43000.0, 'num_q': 43.0}

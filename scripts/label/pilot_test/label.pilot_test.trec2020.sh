#!/bin/bash
device=$1 #TODO: input arg
fb_k=$2 #TODO: input arg
beta=$3 #TODO: input arg
fb_clusters=$4 #TODO: input arg

# 1. ANN Search
queries=data/queries.trec2020.tsv #TODO: custom path
index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
experiment_root=experiments/pilot_test/trec2020 #TODO: custom path
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
rerank_experiment=kmeans.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --checkpoint ${checkpoint} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2020qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
    --fb_k ${fb_k} --beta ${beta} --fb_clusters ${fb_clusters} \
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
qrels=data/pilot_test/label/2020qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2020qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
# fb_k: 10, beta: 1.0, fb_clusters: 10
#* {'nDCG@10': 0.7274661897899061, 'nDCG@25': 0.6608499992583979, 'nDCG@50': 0.6371211765376297, 'nDCG@100': 0.6356750227979504, 'nDCG@200': 0.6577465465242718, 'nDCG@500': 0.6845451145868552, 'nDCG@1000': 0.6948931905000151, 'R(rel=2)@3': 0.18637059317455285, 'R(rel=2)@5': 0.26622804837806996, 'R(rel=2)@10': 0.3809043887962409, 'R(rel=2)@25': 0.5305460617185046, 'R(rel=2)@50': 0.6342442404770194, 'R(rel=2)@100': 0.722593318153712, 'R(rel=2)@200': 0.7807346151848998, 'R(rel=2)@1000': 0.8404711529736575, 'P(rel=2)@3': 0.7654320987654322, 'P(rel=2)@5': 0.6962962962962964, 'P(rel=2)@10': 0.5648148148148148, 'P(rel=2)@25': 0.36740740740740735, 'P(rel=2)@50': 0.2474074074074074, 'P(rel=2)@100': 0.15296296296296297, 'P(rel=2)@200': 0.08666666666666664, 'AP(rel=2)@1000': 0.47268733037997024, 'RR(rel=2)@10': 0.9004629629629629, 'NumRet': 53951.0, 'num_q': 54.0}
# fb_k: 10, beta: 0.5, fb_clusters: 10 #! default
#* {'nDCG@10': 0.7338781105662351, 'nDCG@25': 0.6775293935547142, 'nDCG@50': 0.656621370829082, 'nDCG@100': 0.6534349678345813, 'nDCG@200': 0.6705277830400258, 'nDCG@500': 0.6953974076379872, 'nDCG@1000': 0.7050148455816653, 'R(rel=2)@3': 0.19377674078941282, 'R(rel=2)@5': 0.2892370832259377, 'R(rel=2)@10': 0.39659083990353067, 'R(rel=2)@25': 0.5552675170142956, 'R(rel=2)@50': 0.6600335491285704, 'R(rel=2)@100': 0.7430883884610604, 'R(rel=2)@200': 0.7880887595465768, 'R(rel=2)@1000': 0.8409960558735947, 'P(rel=2)@1': 0.8333333333333334, 'P(rel=2)@3': 0.7716049382716049, 'P(rel=2)@5': 0.7037037037037036, 'P(rel=2)@10': 0.5685185185185185, 'P(rel=2)@25': 0.3807407407407407, 'P(rel=2)@50': 0.2562962962962964, 'P(rel=2)@100': 0.15722222222222218, 'P(rel=2)@200': 0.08777777777777779, 'AP(rel=2)@1000': 0.4947040594180181, 'RR(rel=2)@10': 0.8956790123456789, 'NumRet': 53951.0, 'num_q': 54.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_k: 10, beta: 1.0, fb_clusters: 10
#* {'nDCG@10': 0.7943854644944643, 'nDCG@25': 0.7109220839991847, 'nDCG@50': 0.6821852847390554, 'nDCG@100': 0.676345420342338, 'nDCG@200': 0.6973642363764166, 'nDCG@500': 0.7233631846910504, 'nDCG@1000': 0.7334608017912321, 'R(rel=2)@3': 0.2017555600069262, 'R(rel=2)@5': 0.2855787298396843, 'R(rel=2)@10': 0.4109098404309166, 'R(rel=2)@25': 0.5521756838312053, 'R(rel=2)@50': 0.6513732082289077, 'R(rel=2)@100': 0.7343584410271854, 'R(rel=2)@200': 0.7885903319044186, 'R(rel=2)@1000': 0.843769451781273, 'P(rel=2)@3': 0.8580246913580248, 'P(rel=2)@5': 0.7740740740740739, 'P(rel=2)@10': 0.625925925925926, 'P(rel=2)@25': 0.394074074074074, 'P(rel=2)@50': 0.2637037037037037, 'P(rel=2)@100': 0.16166666666666668, 'P(rel=2)@200': 0.09111111111111109, 'AP(rel=2)@1000': 0.5341867693883938, 'RR(rel=2)@10': 1.0, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_k: 10, beta: 0.5, fb_clusters: 10 #! default
#* {'nDCG@10': 0.7888216616010204, 'nDCG@25': 0.7222633216667409, 'nDCG@50': 0.6946214440874632, 'nDCG@100': 0.6873853239958142, 'nDCG@200': 0.7036284878705854, 'nDCG@500': 0.7277748934431627, 'nDCG@1000': 0.737136922792497, 'R(rel=2)@3': 0.2074647366817113, 'R(rel=2)@5': 0.2912573669700655, 'R(rel=2)@10': 0.40992922432538664, 'R(rel=2)@25': 0.5743102490976296, 'R(rel=2)@50': 0.6746917987193155, 'R(rel=2)@100': 0.7513247850391765, 'R(rel=2)@200': 0.7956864392403469, 'R(rel=2)@1000': 0.8442870262812905, 'P(rel=2)@1': 0.9629629629629629, 'P(rel=2)@3': 0.8518518518518519, 'P(rel=2)@5': 0.7703703703703704, 'P(rel=2)@10': 0.6203703703703703, 'P(rel=2)@25': 0.4103703703703704, 'P(rel=2)@50': 0.2733333333333334, 'P(rel=2)@100': 0.16555555555555557, 'P(rel=2)@200': 0.09222222222222222, 'AP(rel=2)@1000': 0.5457664050358632, 'RR(rel=2)@10': 0.9783950617283951, 'NumRet': 54000.0, 'num_q': 54.0}

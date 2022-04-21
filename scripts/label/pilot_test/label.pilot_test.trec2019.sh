#!/bin/bash
device=$1 #TODO: input arg
fb_k=$2 #TODO: input arg
beta=$3 #TODO: input arg
fb_clusters=$4 #TODO: input arg



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
rerank_experiment=kmeans.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
ranking_jsonl=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.jsonl
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    #TODO: uncomment
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --checkpoint ${checkpoint} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --root ${experiment_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2019qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
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
qrels=data/pilot_test/label/2019qrels-pass.test.tsv #TODO: custom path
qrels_exclude=data/pilot_test/label/2019qrels-pass.train.tsv #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels} --qrels_exclude ${qrels_exclude} --ranking ${ranking} > ${result_path}
echo;cat ${result_path} | tail -1;echo
# fb_k: 10, beta: 1.0, fb_clusters: 10
#* {'nDCG@10': 0.7659607045954312, 'nDCG@25': 0.7201282611080616, 'nDCG@50': 0.6944674781622254, 'nDCG@100': 0.6760465649712458, 'nDCG@200': 0.6805447497347316, 'nDCG@500': 0.7082651907706733, 'nDCG@1000': 0.7184540479652519, 'R(rel=2)@3': 0.13335783259916134, 'R(rel=2)@5': 0.2021457400996973, 'R(rel=2)@10': 0.30092896275172426, 'R(rel=2)@25': 0.45266129933853244, 'R(rel=2)@50': 0.5741926898797947, 'R(rel=2)@100': 0.6600785221474331, 'R(rel=2)@200': 0.7358852806433348, 'R(rel=2)@1000': 0.820168923053885, 'P(rel=2)@3': 0.7596899224806203, 'P(rel=2)@5': 0.7395348837209303, 'P(rel=2)@10': 0.672093023255814, 'P(rel=2)@25': 0.5069767441860463, 'P(rel=2)@50': 0.3832558139534884, 'P(rel=2)@100': 0.2730232558139535, 'P(rel=2)@200': 0.17220930232558143, 'AP(rel=2)@1000': 0.4903017289274749, 'RR(rel=2)@10': 0.8856589147286822, 'NumRet': 42962.0, 'num_q': 43.0}
# fb_k: 10, beta: 0.5, fb_clusters: 10 #! default
#* {'nDCG@10': 0.7763703561062344, 'nDCG@25': 0.743330295323221, 'nDCG@50': 0.723884248893208, 'nDCG@100': 0.7049011609855751, 'nDCG@200': 0.7087503684835873, 'nDCG@500': 0.7304143260658776, 'nDCG@1000': 0.7391958213304132, 'R(rel=2)@3': 0.1692309517064902, 'R(rel=2)@5': 0.21780289212965137, 'R(rel=2)@10': 0.3129772225426723, 'R(rel=2)@25': 0.4751924263348525, 'R(rel=2)@50': 0.5933494983298608, 'R(rel=2)@100': 0.6821178733088188, 'R(rel=2)@200': 0.7585472506653059, 'R(rel=2)@1000': 0.8225489366551176, 'P(rel=2)@1': 0.9069767441860465, 'P(rel=2)@3': 0.8217054263565892, 'P(rel=2)@5': 0.7534883720930231, 'P(rel=2)@10': 0.6627906976744186, 'P(rel=2)@25': 0.518139534883721, 'P(rel=2)@50': 0.39674418604651174, 'P(rel=2)@100': 0.28093023255813954, 'P(rel=2)@200': 0.17837209302325582, 'AP(rel=2)@1000': 0.5355647942720198, 'RR(rel=2)@10': 0.9350775193798451, 'NumRet': 42962.0, 'num_q': 43.0}
# fb_k: 5, beta: 0.5, fb_clusters: 10
#* {'nDCG@10': 0.7620717478960636, 'nDCG@25': 0.7351417388816305, 'nDCG@50': 0.7159015438024591, 'nDCG@100': 0.6996612939505243, 'nDCG@200': 0.7043479144098026, 'nDCG@500': 0.7270609027357511, 'nDCG@1000': 0.7361144555291471, 'R(rel=2)@3': 0.16031293830483467, 'R(rel=2)@5': 0.20895700705257617, 'R(rel=2)@10': 0.30734724094814253, 'R(rel=2)@25': 0.4756235650215733, 'R(rel=2)@50': 0.5868318499031209, 'R(rel=2)@100': 0.6795853462500016, 'R(rel=2)@200': 0.7591882272557477, 'R(rel=2)@1000': 0.8227193089185132, 'P(rel=2)@3': 0.7906976744186046, 'P(rel=2)@5': 0.7255813953488371, 'P(rel=2)@10': 0.6511627906976745, 'P(rel=2)@25': 0.5116279069767441, 'P(rel=2)@50': 0.3911627906976745, 'P(rel=2)@100': 0.2795348837209302, 'P(rel=2)@200': 0.17767441860465116, 'AP(rel=2)@1000': 0.5251497054918531, 'RR(rel=2)@10': 0.9335917312661499, 'NumRet': 42962.0, 'num_q': 43.0}
# 
result_all_path=${experiment_root}/${rerank_experiment}/label.py/$(ls ${experiment_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2019/2019qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_k: 10, beta: 1.0, fb_clusters: 10
#* {'nDCG@10': 0.8007591702421741, 'nDCG@25': 0.7444333191608051, 'nDCG@50': 0.7166247578191156, 'nDCG@100': 0.6965686723953277, 'nDCG@200': 0.6995978518844846, 'nDCG@500': 0.7268478656512178, 'nDCG@1000': 0.7368854137882658, 'R(rel=2)@3': 0.13836464004737772, 'R(rel=2)@5': 0.20440547830902964, 'R(rel=2)@10': 0.30388187407939365, 'R(rel=2)@25': 0.4601825292151858, 'R(rel=2)@50': 0.5788553459032946, 'R(rel=2)@100': 0.6676275556729856, 'R(rel=2)@200': 0.7413482027314663, 'R(rel=2)@1000': 0.823791958188624, 'P(rel=2)@3': 0.8294573643410852, 'P(rel=2)@5': 0.776744186046512, 'P(rel=2)@10': 0.7069767441860466, 'P(rel=2)@25': 0.5283720930232556, 'P(rel=2)@50': 0.39581395348837217, 'P(rel=2)@100': 0.2811627906976744, 'P(rel=2)@200': 0.17627906976744187, 'AP(rel=2)@1000': 0.5179818717722298, 'RR(rel=2)@10': 0.9263565891472869, 'NumRet': 43000.0, 'num_q': 43.0} 
# fb_k: 10, beta: 0.5, fb_clusters: 10 #! default
#* {'nDCG@10': 0.8120135662747803, 'nDCG@25': 0.7703863738264701, 'nDCG@50': 0.7475305066860802, 'nDCG@100': 0.7257067931549842, 'nDCG@200': 0.7282057351251486, 'nDCG@500': 0.7491020607359591, 'nDCG@1000': 0.7577352250713204, 'R(rel=2)@3': 0.17063595227170247, 'R(rel=2)@5': 0.2255188156196239, 'R(rel=2)@10': 0.3128511512438877, 'R(rel=2)@25': 0.4855550731666792, 'R(rel=2)@50': 0.6021510470773905, 'R(rel=2)@100': 0.6885124644456627, 'R(rel=2)@200': 0.7636500642094725, 'R(rel=2)@1000': 0.8261668312407895, 'P(rel=2)@1': 0.9534883720930233, 'P(rel=2)@3': 0.8914728682170542, 'P(rel=2)@5': 0.8139534883720931, 'P(rel=2)@10': 0.6953488372093024, 'P(rel=2)@25': 0.5423255813953488, 'P(rel=2)@50': 0.4116279069767442, 'P(rel=2)@100': 0.28906976744186036, 'P(rel=2)@200': 0.18279069767441858, 'AP(rel=2)@1000': 0.5628715004189373, 'RR(rel=2)@10': 0.9709302325581395, 'NumRet': 43000.0, 'num_q': 43.0}
# fb_k: 5, beta: 0.5, fb_clusters: 10
#* {'nDCG@10': 0.7982767456174364, 'nDCG@25': 0.7597299698682651, 'nDCG@50': 0.7378556248550183, 'nDCG@100': 0.7180770681131352, 'nDCG@200': 0.7219568552393377, 'nDCG@500': 0.7439707425351256, 'nDCG@1000': 0.7528735509862899, 'R(rel=2)@3': 0.16865161649803703, 'R(rel=2)@5': 0.21162623092443003, 'R(rel=2)@10': 0.31524302688240174, 'R(rel=2)@25': 0.48712851540140395, 'R(rel=2)@50': 0.5952199411426032, 'R(rel=2)@100': 0.6855149656101904, 'R(rel=2)@200': 0.763934552898511, 'R(rel=2)@1000': 0.8263341276820402, 'P(rel=2)@3': 0.8759689922480618, 'P(rel=2)@5': 0.7813953488372095, 'P(rel=2)@10': 0.688372093023256, 'P(rel=2)@25': 0.5358139534883721, 'P(rel=2)@50': 0.4060465116279069, 'P(rel=2)@100': 0.2869767441860464, 'P(rel=2)@200': 0.1819767441860465, 'AP(rel=2)@1000': 0.5523042985956906, 'RR(rel=2)@10': 0.9697674418604652, 'NumRet': 43000.0, 'num_q': 43.0}



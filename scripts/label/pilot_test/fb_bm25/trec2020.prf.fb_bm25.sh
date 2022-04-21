#!/bin/bash
device=$1 #TODO: input arg
fb_docs=$2 #TODO: input arg
fb_k=$3 #TODO: input arg
beta=$4 #TODO: input arg
fb_clusters=$5 #TODO: input arg

exp_root=experiments/pilot_test/fb_bm25/trec2020 #TODO: custom path
# mkdir -p ${exp_root}

# 1. ANN Search
queries=data/queries.trec2020.tsv #TODO: custom path
index_root=experiments/colbert.teacher/MSMARCO-psg/index.py #TODO: custom path
checkpoint=data/checkpoints/colbert.teacher.dnn #TODO: custom path
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
[ ! -d "${index_root}" ] && echo "${index_root} does not exist. (Please, index documents first, and then search NN documents for queries.)" && return

ann_experiment=wo_qe #TODO: custom path
ann_exp_root=experiments/pilot_test/trec2020 #TODO: custom path
topk=${ann_exp_root}/${ann_experiment}/retrieve.py/$(ls ${ann_exp_root}/${ann_experiment}/retrieve.py)/unordered.tsv
if [ ! -f ${topk} ];then
    echo "1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small."
    # 1. ANN search (FAISS), using original queries without query expansion: the retrieved document candidates by ANN search achieve near 100% recall for relevant documents, on dev.small.
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries ${queries} \
    --nprobe 32 --partitions 32768 --faiss_depth 512 --index_root ${index_root} --index_name MSMARCO.L2.32x200k \
    --checkpoint ${checkpoint} --root ${ann_exp_root} --experiment ${ann_experiment}
else
    echo "We have ANN search result at: \"${topk}\""
fi



# 2. Exact-NN Search, using query expansion.
topk=${ann_exp_root}/${ann_experiment}/retrieve.py/$(ls ${ann_exp_root}/${ann_experiment}/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
# 
# rerank_experiment=kmeans.prf_only.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
rerank_experiment=prf.fb_bm25.docs${fb_docs}.k${fb_k}.beta${beta}.clusters${fb_clusters} #TODO: custom path
# 
ranking_jsonl=${exp_root}/${rerank_experiment}/label.py/$(ls ${exp_root}/${rerank_experiment}/label.py)/ranking.jsonl
# 
# fb_ranking=experiments/pilot_test/trec2020/wo_qe/label.py/$(ls experiments/pilot_test/trec2020/wo_qe/label.py/)/ranking.jsonl #TODO: custom path
fb_ranking=experiments/pilot_test/fb_bm25/trec2020/docT5query.trec2020.jsonl #TODO: custom path
# 
if [ ! -f ${ranking_jsonl} ];then
    echo "2. Exact-NN Search, using query expansion."
    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --topk ${topk} --batch --log-scores \
    --queries ${queries} --checkpoint ${checkpoint} \
    --index_root ${index_root} --index_name MSMARCO.L2.32x200k --nprobe 32 --partitions 32768 --faiss_depth 1024 \
    --root ${exp_root} --experiment ${rerank_experiment} \
    --qrels data/pilot_test/label/2020qrels-pass.train.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
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
ranking=${exp_root}/${rerank_experiment}/label.py/$(ls ${exp_root}/${rerank_experiment}/label.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m preprocessing.utils.sanity_check.same_qids_in_queries_and_ranked_pids --queries ${queries} --ranking ${ranking}
# 
result_all_path=${exp_root}/${rerank_experiment}/label.py/$(ls ${exp_root}/${rerank_experiment}/label.py)/ranking.metrics.all
qrels_all=data/trec2020/2020qrels-pass.txt #TODO: custom path
python -m utility.evaluate.trec_passages --qrels ${qrels_all} --ranking ${ranking} > ${result_all_path}
echo;cat ${result_all_path} | tail -1;echo
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7143217834958663, 'nDCG@25': 0.6710491158642045, 'nDCG@50': 0.6574003919278679, 'nDCG@100': 0.6603080817198829, 'nDCG@200': 0.6809886728578028, 'nDCG@500': 0.704690587518724, 'nDCG@1000': 0.7128823667835437, 'R(rel=2)@3': 0.16219987231208802, 'R(rel=2)@5': 0.2788205957591118, 'R(rel=2)@10': 0.3936454405929331, 'R(rel=2)@25': 0.5535927287458687, 'R(rel=2)@50': 0.6549083439586871, 'R(rel=2)@100': 0.7515129872142341, 'R(rel=2)@200': 0.7971908280777683, 'R(rel=2)@1000': 0.8470493803792577, 'P(rel=2)@1': 0.7592592592592593, 'P(rel=2)@3': 0.7037037037037034, 'P(rel=2)@5': 0.6814814814814816, 'P(rel=2)@10': 0.561111111111111, 'P(rel=2)@25': 0.374074074074074, 'P(rel=2)@50': 0.2611111111111111, 'P(rel=2)@100': 0.16574074074074072, 'P(rel=2)@200': 0.09277777777777778, 'P(rel=2)@1000': 0.021037037037037045, 'AP(rel=2)@1000': 0.48551608842808264, 'RR(rel=2)@10': 0.8412257495590829, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.6948483672295324, 'nDCG@25': 0.6513205207286128, 'nDCG@50': 0.6385349170940494, 'nDCG@100': 0.6411867957435261, 'nDCG@200': 0.6653074060601818, 'nDCG@500': 0.6904451590423245, 'nDCG@1000': 0.6994749043939502, 'R(rel=2)@3': 0.1560167732737398, 'R(rel=2)@5': 0.26175624881992954, 'R(rel=2)@10': 0.3923477092656577, 'R(rel=2)@25': 0.5397921113288819, 'R(rel=2)@50': 0.6498035998368864, 'R(rel=2)@100': 0.7325783881902774, 'R(rel=2)@200': 0.788821539010132, 'R(rel=2)@1000': 0.847791792636417, 'P(rel=2)@1': 0.6666666666666666, 'P(rel=2)@3': 0.6975308641975309, 'P(rel=2)@5': 0.6592592592592594, 'P(rel=2)@10': 0.5537037037037036, 'P(rel=2)@25': 0.36296296296296293, 'P(rel=2)@50': 0.25703703703703706, 'P(rel=2)@100': 0.1622222222222222, 'P(rel=2)@200': 0.09259259259259262, 'P(rel=2)@1000': 0.021018518518518527, 'AP(rel=2)@1000': 0.46125859989221063, 'RR(rel=2)@10': 0.7870370370370371, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 1, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.6810724615487479, 'nDCG@25': 0.6571738288972396, 'nDCG@50': 0.6442379940606777, 'nDCG@100': 0.6457556817109947, 'nDCG@200': 0.6648858140234603, 'nDCG@500': 0.6883772821893103, 'nDCG@1000': 0.6991371130068011, 'R(rel=2)@3': 0.15968403499294268, 'R(rel=2)@5': 0.26846982698784444, 'R(rel=2)@10': 0.3836818260379076, 'R(rel=2)@25': 0.5634743703478695, 'R(rel=2)@50': 0.6628707237884918, 'R(rel=2)@100': 0.740641417585591, 'R(rel=2)@200': 0.791670176847015, 'R(rel=2)@1000': 0.8441678789577145, 'P(rel=2)@1': 0.6296296296296297, 'P(rel=2)@3': 0.6728395061728394, 'P(rel=2)@5': 0.6703703703703703, 'P(rel=2)@10': 0.5370370370370369, 'P(rel=2)@25': 0.37999999999999995, 'P(rel=2)@50': 0.2633333333333333, 'P(rel=2)@100': 0.1627777777777778, 'P(rel=2)@200': 0.0915740740740741, 'P(rel=2)@1000': 0.020833333333333343, 'AP(rel=2)@1000': 0.47168227165106047, 'RR(rel=2)@10': 0.7743827160493828, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 1, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.673435657908828, 'nDCG@25': 0.6422056514504185, 'nDCG@50': 0.6361769059454921, 'nDCG@100': 0.6343654407827842, 'nDCG@200': 0.6534380047118136, 'nDCG@500': 0.6795563849871742, 'nDCG@1000': 0.690861877538559, 'R(rel=2)@3': 0.1531837837853956, 'R(rel=2)@5': 0.25162836889265466, 'R(rel=2)@10': 0.3771611189114733, 'R(rel=2)@25': 0.5323462992116057, 'R(rel=2)@50': 0.652913971814638, 'R(rel=2)@100': 0.7273401633863236, 'R(rel=2)@200': 0.7801100271988014, 'R(rel=2)@1000': 0.8405036418802403, 'P(rel=2)@1': 0.6296296296296297, 'P(rel=2)@3': 0.6419753086419753, 'P(rel=2)@5': 0.6333333333333331, 'P(rel=2)@10': 0.5425925925925925, 'P(rel=2)@25': 0.3659259259259258, 'P(rel=2)@50': 0.2596296296296296, 'P(rel=2)@100': 0.16074074074074074, 'P(rel=2)@200': 0.08953703703703704, 'P(rel=2)@1000': 0.02081481481481482, 'AP(rel=2)@1000': 0.4563930508771226, 'RR(rel=2)@10': 0.7592592592592593, 'NumRet': 54000.0, 'num_q': 54.0}



# ColBERT
#* {'nDCG@10': 0.6746415559920877, 'nDCG@25': 0.6408579361254387, 'nDCG@50': 0.6284356710611297, 'nDCG@100': 0.6296520709973429, 'nDCG@200': 0.6482406927534247, 'nDCG@500': 0.6738405847608164, 'nDCG@1000': 0.6862852044096319, 'R(rel=2)@3': 0.16003239577496967, 'R(rel=2)@5': 0.2476104839708578, 'R(rel=2)@10': 0.38692970913678476, 'R(rel=2)@25': 0.5430444561689066, 'R(rel=2)@50': 0.65296216952532, 'R(rel=2)@100': 0.7291550798979732, 'R(rel=2)@200': 0.7767769930691577, 'R(rel=2)@1000': 0.8425068725529834, 'P(rel=2)@1': 0.7037037037037037, 'P(rel=2)@3': 0.6543209876543209, 'P(rel=2)@5': 0.6111111111111113, 'P(rel=2)@10': 0.5296296296296296, 'P(rel=2)@25': 0.3651851851851853, 'P(rel=2)@50': 0.2522222222222222, 'P(rel=2)@100': 0.1562962962962963, 'P(rel=2)@200': 0.08833333333333335, 'AP(rel=2)@1000': 0.4637476631143032, 'RR(rel=2)@10': 0.8140432098765432, 'NumRet': 54000.0, 'num_q': 54.0}
# 
# ColBERT-PRF
# fb_docs: 3, fb_k: 10, beta: 0.5, fb_clusters: 24
#* {'nDCG@10': 0.7050928028350049, 'nDCG@25': 0.6617098002216392, 'nDCG@50': 0.643111928388147, 'nDCG@100': 0.645560913726691, 'nDCG@200': 0.6700730087032704, 'nDCG@500': 0.6944169843121778, 'nDCG@1000': 0.7045979669463018, 'R(rel=2)@3': 0.16461119157641396, 'R(rel=2)@5': 0.277456583645939, 'R(rel=2)@10': 0.4064567772014026, 'R(rel=2)@25': 0.5584931435597346, 'R(rel=2)@50': 0.6512125062199494, 'R(rel=2)@100': 0.7334187942894212, 'R(rel=2)@200': 0.7914281738072042, 'R(rel=2)@1000': 0.8471683391316858, 'P(rel=2)@3': 0.6851851851851853,'P(rel=2)@5': 0.6592592592592593, 'P(rel=2)@10': 0.5518518518518517, 'P(rel=2)@25': 0.3725925925925926, 'P(rel=2)@50': 0.25037037037037035, 'P(rel=2)@100': 0.1596296296296296, 'P(rel=2)@200': 0.09240740740740741, 'AP(rel=2)@1000': 0.4876691034852261, 'RR(rel=2)@10': 0.8339506172839507, 'NumRet': 54000.0, 'num_q': 54.0}
# fb_docs: 3, fb_k: 10, beta: 1.0, fb_clusters: 24
#* {'nDCG@10': 0.6885155204680559, 'nDCG@25': 0.6459207218472783, 'nDCG@50': 0.6251576248660592, 'nDCG@100': 0.6267060994294082, 'nDCG@200': 0.6568601222189001, 'nDCG@500': 0.6842351267567333, 'nDCG@1000': 0.6942496874574772, 'R(rel=2)@3': 0.15843835207024112, 'R(rel=2)@5': 0.27599721670000044, 'R(rel=2)@10': 0.38921134194923446, 'R(rel=2)@25': 0.5396545429141235, 'R(rel=2)@50': 0.6308315789493557, 'R(rel=2)@100': 0.7058827089149365, 'R(rel=2)@200': 0.7840546573129827, 'R(rel=2)@1000': 0.8461340321277788, 'P(rel=2)@3': 0.6790123456790123, 'P(rel=2)@5': 0.6666666666666669, 'P(rel=2)@10': 0.5388888888888889, 'P(rel=2)@25': 0.3599999999999998, 'P(rel=2)@50': 0.2414814814814815, 'P(rel=2)@100': 0.15333333333333332, 'P(rel=2)@200': 0.09120370370370372, 'AP(rel=2)@1000': 0.46920678181262254, 'RR(rel=2)@10': 0.8240740740740742, 'NumRet': 54000.0, 'num_q': 54.0}

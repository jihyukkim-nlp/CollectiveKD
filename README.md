# CollectiveKD

## Data format

You need the following data.

For training:
- passage collection: `collection.tsv`
- query collection for training dataset: `queries.train.tsv`
- training triples (query, positive passage, negative passage): `triples.train.small.ids.jsonl`

For validation:
- validation triples (query, positive passage, negative passage): `top1000.dev` 
- query collection for validation dataset: `queries.dev.small.tsv`
- relevance annotation for validation dataset: `qrels.dev.small.tsv`


**[1] `collection.tsv`**

Example)
```
0       The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.
1       The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.
2       Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade.
3       The Manhattan Project was the name for a project conducted during World War II, to develop the first atomic bomb. It refers specifically to the period of the project from 194 â¦ 2-1946 under the control of the U.S. Army Corps of Engineers, under the administration of General Leslie R. Groves.
4       versions of each volume as well as complementary websites. The first websiteâThe Manhattan Project: An Interactive Historyâis available on the Office of History and Heritage Resources website, http://www.cfo. doe.gov/me70/history. The Office of History and Heritage Resources and the National Nuclear Security
```

|column     |data type  | description |
|:----      |:----      |:----        |
|1          |int        | passage id (pid) |
|2          |str        | passage |


**[2] `queries.train.tsv` or `queries.dev.small.tsv`**

Examples from `queries.train.tsv`)
```
121352  define extreme
634306  what does chattel mean on credit history
920825  what was the great leap forward brainly
510633  tattoo fixers how much does it cost
737889  what is decentralization process.
```

|column     |data type  | description |
|:----      |:----      |:----        |
|1          |int        | query id (qid) |
|2          |str        | query|


**[3] `triples.train.small.ids.jsonl`**

Example)
```
[400296, 1540783, 3518497]
[662731, 193249, 2975302]
[238256, 4435042, 100008]
[527862, 1505983, 2975302]
[275813, 5736515, 1238670]
[984152, 2304924, 3372067]
[294432, 2592502, 2592504]
[444656, 2932850, 2975302]
[81644, 1097740, 2747766]
[189845, 1051356, 4238671]
```
|column   |data type  | description |
|:----      |:----      |:----        |
|1          |int        | qid (ID in `queries.train.tsv`)  |
|2          |int        | pid of positve passage (ID in `collection.tsv`)|
|3          |int        | pid of negative passage (ID in `collection.tsv`) |

**How positive/negative passages were obtained.**
- The *positive* passages are labeled by human annotators.
- The *negative* passages are sampled from unlabeled passages.


**[4] `top1000.dev`**

Example)
```
188714  1000052 foods and supplements to lower blood sugar      Watch portion sizes: _ Even healthy foods will cause high blood sugar if you eat too much. _ Make sure each of your meals has the same amount of CHOs. Avoid foods high in sugar: _ Some foods to avoid: sugar, honey, candies, syrup, cakes, cookies, regular soda and.
1082792 1000084 what does the golgi apparatus do to the proteins and lipids once they arrive ?  Start studying Bonding, Carbs, Proteins, Lipids. Learn vocabulary, terms, and more with flashcards, games, and other study tools.
995526  1000094 where is the federal penitentiary in ind        It takes THOUSANDS of Macy's associates to bring the MAGIC of MACY'S to LIFE! Our associate team is an invaluable part of who we are and what we do. F ind the seasonal job that's right for you at holiday.macysJOBS.com!
199776  1000115 health benefits of eating vegetarian    The good news is that you will discover what goes into action spurs narrowing of these foods not only a theoretical supposition there are diagnosed with great remedy is said that most people and more can be done. Duncan was a wonderful can eating chicken cause gout benefits of natural. options with your health.
660957  1000115 what foods are good if you have gout?   The good news is that you will discover what goes into action spurs narrowing of these foods not only a theoretical supposition there are diagnosed with great remedy is said that most people and more can be done. Duncan was a wonderful can eating chicken cause gout benefits of natural. options with your health.
```
|column|data type  | description |
|:---- |:----      |:----        |
|1     |int        | qid (ID in `queries.train.tsv`)  |
|2     |int        | pid of positve passage (ID in `collection.tsv`)|
|3     |str        | positive passage|
|4     |str        | negative passage|


**[5] `qrels.dev.small.tsv`**

Example)
```
300674  0       7067032 1
125705  0       7067056 1
94798   0       7067181 1
9083    0       7067274 1
174249  0       7067348 1
```

|column     |data type  | description   |
|:----      |:----      |:----          |
|1          |int        | qid           |
|2          |int        | 0 (dummy; ignore this)     |
|3          |int        | pid of relevant passage (ID in `collection.tsv`)          |
|4          |int        | 1 (dummy; ignore this)     |



## Step 1. Preliminary: Training ColBERT Teacher

- The checkpoint we used in our experiments can be downloaded [here (colbert.dnn)](https://drive.google.com/drive/folders/1Bk6-7KVl6bTDc-2cBtxh7PF7FNSEPjpL?usp=sharing)

An example bash command for **training**:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 29500 \
-m colbert.train --maxsteps 400000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples /path/to/triples.jsonl \
--queries /path/to/queries.train.tsv \
--collection /path/to/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root experiments/colbert-b36-lr3e6 --experiment MSMARCO-psg --run msmarco.psg.l2
```

An example bash command for **validation** (on re-ranking task):
```bash
checkpoint_to_be_validated=experiments/colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-100000.dnn
checkpoint_to_be_validated=experiments/colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn
checkpoint_to_be_validated=experiments/colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-300000.dnn
checkpoint_to_be_validated=experiments/colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-400000.dnn
CUDA_VISIBLE_DEVICES=0 \
python -m colbert.test --checkpoint ${checkpoint_to_be_validated} \
--amp --doc_maxlen 180 --mask-punctuation \
--collection /path/to/collection.tsv \
--queries /path/to/queries.dev.small.tsv \
--qrels /path/to/qrels.dev.small.tsv \
--topk /path/to/top1000.dev \
--root experiments/colbert-b36-lr3e6 --experiment MSMARCO-psg
```



## Step 2. Obtain Collective Knowledge from Pseudo-Relevance Feedback (PRF) Passages 

The overall process is as follows:
1. encoding and indexing.
2. retrieval, to obtain pseudo-relevance feedback (PRF).
3. obtaining collective knowledge from PRF.

**Step 2-1: Encoding and Indexing.**

An example bash command for **encoding and indexing**:
```bash
# encoding passages
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 \
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 30000 \
-m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--checkpoint /path/to/colbert.dnn --collection /path/to/collection.tsv \
--index_root experiments/colbert-b36-lr3e6/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k \
--root experiments/colbert-b36-lr3e6 --experiment MSMARCO-psg

# faiss indexing for approximate nearest neighbor search
CUDA_VISIBLE_DEVICES=0,1 python -m colbert.index_faiss \
--index_root experiments/colbert-b36-lr3e6/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 --slices 1 \
--root experiments/colbert-b36-lr3e6 --experiment MSMARCO-psg
```

**Step 2-2: Retrieval.**

- The ranking file for training queries, we used in our experiments as PRF, can be downloaded [here (colbert.msmarco_pass.train.ranking.jsonl)](https://drive.google.com/drive/folders/1YQzYKgY7uioSiUxVPgBIf4Ax3sG-mFTI?usp=sharing)

scp dilab4:/data1/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/colbert.teacher/MSMARCO-psg-HN/label.py/ranking.jsonl ./colbert.msmarco_pass.train.ranking.jsonl

An example bash command for **retrieval, to obtain pseudo-relevance feedback (PRF)**:
``` bash
echo;echo;echo
# 1. ANN search (FAISS)
topk_dir=experiments/colbert-b36-lr3e6/MSMARCO-psg-PRF/retrieve.py/pseudo_relevance_feedback
topk=${topk_dir}/unordered.tsv
if [ ! -f ${topk} ];then
    CUDA_VISIBLE_DEVICES=${device} python -m colbert.retrieve --batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --queries /path/to/queries \
    --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root experiments/colbert-b36-lr3e6/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k \
    --checkpoint /path/to/colbert.dnn --root experiments/colbert-b36-lr3e6 --experiment MSMARCO-psg-PRF --run pseudo_relevance_feedback
else
    echo "We have ANN search result at: \"${topk}\""
fi

echo;echo;echo
# 2. Split the large query file into small files, to prevent out-of-memory
[ ! -f "${queries}" ] && echo "${queries} does not exist." && return
queries_split=/path/to/queries.train.splits
echo "1. Split the large query file into small files, to prevent out-of-memory"
echo "mkdir ${queries_split}"
mkdir -p ${queries_split}
echo "split \"${queries}\" into multiple queries with 100000 lines each"
split -d -l 50000 ${queries} ${queries_split}/queries.tsv.
echo "Splitted query files"
wc -l ${queries_split}/*
n_splits=$(ls ${queries_split} | wc -l)
echo


echo;echo;echo
# 3. Filter ANN search result (top-K pids in ``unordered.tsv``), using each split queries
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
topk_split=${topk_dir}/queries.train.splits #TODO: custom path
echo "3. Split the large unordered.tsv file into small files, to prevent out-of-memory"
echo "mkdir ${topk_split}"
mkdir -p ${topk_split}
small_queries=""
filtered_topk=""
for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
    small_queries="${small_queries} ${queries_split}/queries.tsv.${i}"
    filtered_topk="${filtered_topk} ${topk_split}/unordered.${i}.tsv"
done
python -m preprocessing.utils.filter_topK_pids --topk ${topk} \
--queries ${small_queries} \
--filtered_topk ${filtered_topk}


echo;echo;echo
# 4. Exact-NN search
echo "4. Exact-NN search"
# 
for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
    small_queries=${queries_split}/queries.tsv.${i}
    small_topk=${topk_split}/unordered.${i}.tsv
    [ ! -f "${small_queries}" ] && echo "${small_queries} does not exist." && return
    [ ! -f "${small_topk}" ] && echo "${small_topk} does not exist." && return
    [ ! -d "${index_root}" ] && echo "${index_root} does not exist." && return
    [ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
    [ ! -d "${exp_root}" ] && echo "${exp_root} does not exist." && return

    CUDA_VISIBLE_DEVICES=${device} python \
    -m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
    --batch --log-scores \
    --topk ${small_topk} --queries ${small_queries} \
    --index_root experiments/colbert-b36-lr3e6/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k \
    --checkpoint /path/to/colbert.dnn \
    --qrels /path/to/qrels.train.tsv \
    --collection /path/to/collection.tsv \
    --root experiments/colbert-b36-lr3e6 --experiment MSMARCO-psg-PRF  --run pseudo_relevance_feedback.${i} \
    --fb_k 0 --beta 0.0 --depth 1000 --score_by_range
done


echo;echo;echo
# 5. Merge results
ranking=experiments/colbert-b36-lr3e6/MSMARCO-psg-PRF/label.py/ranking.tsv
ranking_jsonl=experiments/colbert-b36-lr3e6/MSMARCO-psg-PRF/label.py/ranking.jsonl
echo "5. Merge results"
echo -n "" > ${ranking}
echo -n "" > ${ranking_jsonl}
for i in $(seq -f "%02g" 0 $(expr ${n_splits} - 1));do
    small_ranking=experiments/colbert-b36-lr3e6/MSMARCO-psg-PRF/label.py/pseudo_relevance_feedback.${i}/ranking.tsv
    cat ${small_ranking} >> ${ranking}
    small_ranking_jsonl=experiments/colbert-b36-lr3e6/MSMARCO-psg-PRF/label.py/pseudo_relevance_feedback.${i}/ranking.jsonl
    cat ${small_ranking_jsonl} >> ${ranking_jsonl}
    # delete splited file results
    rm -v experiments/colbert-b36-lr3e6/MSMARCO-psg-PRF/label.py/pseudo_relevance_feedback.${i}/ranking.*
done
```

**Step 2-3: Obtaining Collective Knowledge.**

- The collective knowledge from PRF (docs=3, clusters=24, k=10, beta=1.0), we used in our experiments, can be downloaded [here (colbert.msmarco_pass.train.collective_knowledge.pt)](https://drive.google.com/drive/folders/1YQzYKgY7uioSiUxVPgBIf4Ax3sG-mFTI?usp=sharing)
scp dilab4:/data1/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/colbert.teacher/MSMARCO-psg-CollectiveFeedback/docs3.clusters24.k10.beta1.0/label.py/2022-01-04_21.17.26/expansion.pt ./colbert.msmarco_pass.train.collective_knowledge.pt

An example bash command for **obtaining collective knowledge**:
``` bash
CUDA_VISIBLE_DEVICES=0 python \
-m colbert.label --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--root experiments/colbert-b36-lr3e6 --experiment MSMARCO-psg-CollectiveKnowledge \
--expansion_only --prf --fb_docs 3 --fb_k 10 --beta 1.0 --fb_clusters 24 \
--index_root experiments/colbert-b36-lr3e6/MSMARCO-psg/index.py --index_name MSMARCO.L2.32x200k  --nprobe 32 --partitions 32768 --faiss_depth 1024 \
--batch --log-scores \
--fb_ranking /path/to/colbert.msmarco_pass.train.ranking.jsonl \
--checkpoint /path/to/colbert.dnn \
--queries /path/to/queries.train.tsv \
--qrels /path/to/qrels.train.tsv \
--collection /path/to/collection.tsv \
```


## Step 3. Train PQA-ColBERT (pseudo-query-aware ColBERT) w/ ColBERT teacher

### Step 3-1. Fine-tune PQA-ColBERT using BM25 negatives

We adopt the knowledge distillation training framework, similar to [TCT-ColBERT](https://aclanthology.org/2021.repl4nlp-1.17.pdf).
Different from TCT-ColBERT, our student is PQA-ColBERT that uses pruned tokens, while the teacher is the original ColBERT that uses all tokens as in TCT-ColBERT.

An example bash command for training:
```bash
devices=0,1 # GPU devices used for training
n_devices=$(echo ${devices} | awk -F "," '{ print NF }')
echo "n_devices: ${n_devices}" # e.g., "2"

pruned_index_size=65 # The number of remaining tokens after pruning.

pruner_filepath=/path/to/collection.pruner.tsv # e.g., "experiments/qd_pruner/pruner/prune.py/collection.pruner.tsv"
[ ! -f ${pruner_filepath} ] && echo "${pruner_filepath} does not exist" && return
teacher_checkpoint=/path/to/colbert-*.dnn # A checkpoint of ColBERT teacher that uses all document tokens for encoding, e.g., "experiments/colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-300000.dnn"
[ ! -f ${teacher_checkpoint} ] && echo "${teacher_checkpoint} does not exist" && return
triples=/path/to/triples.jsonl # e.g., "data/triples.train.small.ids.jsonl"
[ ! -f ${triples} ] && echo "${triples} does not exist" && return
queries=/path/to/queries.train.tsv # e.g., "data/queries.train.tsv"
[ ! -f ${queries} ] && echo "${queries} does not exist" && return
collection=/path/to/collection.tsv # e.g., "data/collection.tsv"
[ ! -f ${collection} ] && echo "${collection} does not exist" && return

CUDA_VISIBLE_DEVICES=${devices} \
python -m torch.distributed.launch --nproc_per_node=${n_devices} --master_addr 127.0.0.1 --master_port 29500 \
-m colbert.train --maxsteps 200000 --amp --bsize 36 --lr 3e-06 --accum 1 \
--triples ${triples} \
--queries ${queries} --collection ${collection} \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root experiments/pqa_colbert-b36-lr3e6 --experiment MSMARCO-psg --run msmarco.psg.l2 \
--teacher_checkpoint ${teacher_checkpoint} --knowledge_distillation --kd_temperature 0.25 \
--prune_tokens --pruner_filepath ${pruner_filepath} --pruned_index_size ${pruned_index_size} --pseudo_query_indicator
```

An example bash command for validation (on re-ranking task):
```bash
#!/bin/bash
pruned_index_size=65 # The number of remaining tokens after pruning.

checkpoint=/path/to/colbert-*.dnn # A checkpoint for validation, e.g., "experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn"
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return

pruner_filepath=/path/to/collection.pruner.tsv # e.g., "experiments/qd_pruner/pruner/prune.py/collection.pruner.tsv"
[ ! -f ${pruner_filepath} ] && echo "${pruner_filepath} does not exist" && return
queries=/path/to/queries.train.tsv # e.g., "data/queries.train.tsv"
[ ! -f ${queries} ] && echo "${queries} does not exist" && return
collection=/path/to/collection.tsv # e.g., "data/collection.tsv"
[ ! -f ${collection} ] && echo "${collection} does not exist" && return
qrels=/path/to/qrels.dev.small.tsv # e.g., "data/msmarco-pass/qrels.dev.small.tsv"
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return
topk=/path/to/top1000.dev # e.g., "data/msmarco-pass/top1000.dev"
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return

CUDA_VISIBLE_DEVICES=0 \
python -m colbert.test --checkpoint ${checkpoint} \
--prune_tokens --pruner_filepath ${pruner_filepath} --pruned_index_size ${pruned_index_size} --pseudo_query_indicator \
--amp --doc_maxlen 180 --mask-punctuation \
--collection ${collection} --queries ${queries} --qrels ${qrels} --topk ${topk} \
--root experiments/pqa_colbert-b36-lr3e6 --experiment MSMARCO-psg
```

An example bash command for evaluation (on end-to-end ranking task):
- Step 1. indexing: indexing documents using FAISS. 
```bash
#!/bin/bash
pruned_index_size=65 # The number of remaining tokens after pruning.

checkpoint=/path/to/colbert-*.dnn # The best checkpoint, e.g., "experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn"
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
pruner_filepath=/path/to/collection.pruner.tsv # e.g., "experiments/qd_pruner/pruner/prune.py/collection.pruner.tsv"
[ ! -f ${pruner_filepath} ] && echo "${pruner_filepath} does not exist" && return
collection=/path/to/collection.tsv # e.g., "data/collection.tsv"
[ ! -f ${collection} ] && echo "${collection} does not exist" && return

index_root=experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/index.py

# index
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 40000 \
-m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--prune_tokens --pruner_filepath ${pruner_filepath} --pruned_index_size ${pruned_index_size} --pseudo_query_indicator \
--checkpoint ${checkpoint} --collection ${collection} \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--root experiments/pqa_colbert-b36-lr3e6 --experiment MSMARCO-psg \

# index_faiss
CUDA_VISIBLE_DEVICES=0,1 python -m colbert.index_faiss \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 --slices 1 \
--root experiments/pqa_colbert-b36-lr3e6 --experiment MSMARCO-psg
```

- Step 2. ANN search: approximate nearnest neighbor (NN) search using FAISS.
In this step, we retrieve documents that has at least one NN token to any query token.

```bash
checkpoint=/path/to/colbert-*.dnn # The best checkpoint, e.g., "experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn"
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
index_root=/path/to/index.py # e.g., "experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/index.py"
[ ! -d "${index_root}/" ] && echo "${index_root} does not exist." && return
queries=/path/to/queries.train.tsv # e.g., "data/queries.train.tsv"
[ ! -f ${queries} ] && echo "${queries} does not exist" && return

# ANN search
CUDA_VISIBLE_DEVICES=0 python -m colbert.retrieve \
--batch --retrieve_only --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--queries ${queries} \
--nprobe 32 --partitions 32768 --faiss_depth 1024 \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--checkpoint ${checkpoint} --root experiments/pqa_colbert-b36-lr3e6 --experiment MSMARCO-psg

```

- Step 3. Exact-NN search (similar to reranking):
In this step, we rank documents retrieved in Step 2 (ANN search).
```bash
checkpoint=/path/to/colbert-*.dnn # The best checkpoint, e.g., "experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn"
[ ! -f "${checkpoint}" ] && echo "${checkpoint} does not exist." && return
index_root=/path/to/index.py # e.g., "experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/index.py"
[ ! -d "${index_root}/" ] && echo "${index_root} does not exist." && return
queries=/path/to/queries.train.tsv # e.g., "data/queries.train.tsv"
[ ! -f ${queries} ] && echo "${queries} does not exist" && return

topk=experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/retrieve.py/$(ls experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/retrieve.py)/unordered.tsv
[ ! -f "${topk}" ] && echo "${topk} does not exist." && return
CUDA_VISIBLE_DEVICES=0 python -m colbert.rerank --topk ${topk} --batch --log-scores --amp --doc_maxlen 180 --mask-punctuation --bsize 512 \
--queries ${queries} \
--index_root ${index_root} --index_name MSMARCO.L2.32x200k \
--checkpoint ${checkpoint} --root experiments/pqa_colbert-b36-lr3e6 --experiment MSMARCO-psg
```
- Step 4. Evaluation: We evaluate the ranking results, measuring MRR@10/100, Recall@50/200/1000, NDCG@10, MAP@1000
```bash
qrels=/path/to/qrels.dev.small.tsv # e.g., "data/msmarco-pass/qrels.dev.small.tsv"
[ ! -f "${qrels}" ] && echo "${qrels} does not exist." && return
ranking=experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/rerank.py/$(ls experiments/pqa_colbert-b36-lr3e6/MSMARCO-psg/rerank.py)/ranking.tsv
[ ! -f "${ranking}" ] && echo "${ranking} does not exist." && return
python -m utility.evaluate.msmarco_passages --qrels ${qrels} --ranking ${ranking}
```


### Step 3-2. Fine-tune PQA-ColBERT using hard negatives

NAVER dataset 에서는 생략.
Resource 도 많이 잡아먹고, 시간도 너무 오래 걸림.

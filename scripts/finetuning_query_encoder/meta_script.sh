
#* docT5query searching feedback documents

# https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md
# https://github.com/castorini/docTTTTTquery

# This was done in dilab003
cd /hdd/jihyuk/DataCenter/MSMARCO/docT5query
# default BM25, setting k1=0.82 and b=0.68
sh /workspace/GitHubCodes/anserini/target/appassembler/bin/SearchMsmarco -index lucene-index-msmarco-passage-expanded \
-hits 10 -threads 8 \
-queries /workspace/DataCenter/MSMARCO/queries.train.reduced.tsv  -output run.msmarco-passage-expanded.train.reduced.txt
# Total retrieval time: 697.774 s for 502939 queries = 0.001387392904507 per query
# 
sh /workspace/GitHubCodes/anserini/target/appassembler/bin/SearchMsmarco -index lucene-index-msmarco-passage-expanded \
-hits 10 -threads 8 \
-queries /workspace/DataCenter/MSMARCO/queries.dev.small.tsv  -output run.msmarco-passage-expanded.dev.small.txt
# Total retrieval time: 50.930 s: for 6980 queries = 0.007296561604585 per query
sh /workspace/GitHubCodes/anserini/target/appassembler/bin/SearchMsmarco -index lucene-index-msmarco-passage-expanded \
-hits 10 -threads 8 \
-queries /workspace/DataCenter/MSMARCO/queries.trec2019.tsv  -output run.msmarco-passage-expanded.trec2019.pass.txt
# Total retrieval time: 1.309 s: for 43 queries = 0.030441860465116 per query
# 
sh /workspace/GitHubCodes/anserini/target/appassembler/bin/SearchMsmarco -index lucene-index-msmarco-passage-expanded \
-hits 10 -threads 8 \
-queries /workspace/DataCenter/MSMARCO/queries.trec2020.tsv  -output run.msmarco-passage-expanded.trec2020.pass.txt
# Total retrieval time: 1.423 s: for 54 queries = 0.026351851851852 per query
# 
# copy to data
mkdir -p data/fb_docs/docT5query/ranking
cp /workspace/DataCenter/MSMARCO/docT5query/run.msmarco-passage-expanded.* data/fb_docs/docT5query/ranking/
# convert tsv file into json file, for compatibility
python -m scripts.label.pilot_test.fb_bm25.tsv_to_jsonl --tsv data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.train.reduced.txt --output data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.train.reduced.jsonl
python -m scripts.label.pilot_test.fb_bm25.tsv_to_jsonl --tsv data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.dev.small.txt --output data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.dev.small.jsonl
python -m scripts.label.pilot_test.fb_bm25.tsv_to_jsonl --tsv data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2019.pass.txt --output data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2019.pass.jsonl
python -m scripts.label.pilot_test.fb_bm25.tsv_to_jsonl --tsv data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2020.pass.txt --output data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2020.pass.jsonl
# output: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.train.reduced.jsonl
# output: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.dev.small.jsonl
# output: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2019.pass.jsonl
# output: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2020.pass.jsonl
# 
# copy to dilab4, sonic
scp -P 7777 -r data/fb_docs/docT5query/ranking sonic@dilab4.snu.ac.kr:/data1/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/data/fb_docs/docT5query/ranking
scp -P 7777 -r data/fb_docs/docT5query/ranking sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/data/fb_docs/docT5query/ranking
# scp jihyuk@dilab003.yonsei.ac.kr:/hdd/jihyuk/DataCenter/MSMARCO/docT5query/docT5query.queries.train.reduced.tsv experiments/ensemble/fb_docs/
# scp jihyuk@dilab003.yonsei.ac.kr:/hdd/jihyuk/DataCenter/MSMARCO/docT5query/docT5query.queries.dev.small.tsv experiments/ensemble/fb_docs/



# python -m preprocessing.finetuning_query_encoder.triples_uniq_pids --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl
# # of uniq pids: 7445082
# python -m preprocessing.finetuning_query_encoder.triples_uniq_pids --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn4.jsonl
# # of uniq pids: 7731947

# mkdir -p experiments/ensemble-loss/fb_docs/
# scp -P 7777 sonic@dilab4.snu.ac.kr:/data1/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/ensemble-loss/fb_docs/docT5query.queries.train.reduced.* experiments/ensemble-loss/fb_docs/
# # result: experiments/ensemble-loss/fb_docs/docT5query.queries.train.reduced.jsonl
# # 121352  [[2912791, 9.0], [1913621, 8.0], [40219, 7.0], [5561502, 6.0], [40214, 5.0], [7480161, 4.0], [7282917, 3.0], [2912786, 2.0], [5561504, 1.0], [160758, 0.0]]
# # result: experiments/ensemble-loss/fb_docs/docT5query.queries.train.reduced.tsv
# # 121352  2912791 1
# # 121352  1913621 2
# # 121352  40219   3
# # 121352  5561502 4
# # 121352  40214   5
# # 121352  7480161 6
# # 121352  7282917 7
# # 121352  2912786 8
# # 121352  5561504 9
# # 121352  160758  10
# # 634306  1114901 1
# # 634306  3494435 2
# # 634306  1874781 3
# # 634306  1946355 4
# # 634306  8028990 5


#* Query Expansion
# ranking: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.train.reduced.jsonl
python -m preprocessing.feedback_documents.docT5query_expansion \
--collection_jsonl /workspace/DataCenter/MSMARCO/docT5query/msmarco-passage-expanded \
--tfidf_model data/fb_docs/docT5query/tfidf.model.pt \
--fb_docs 3 --fb_k 32 \
--queries data/queries.train.reduced.tsv \
--fb_ranking data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.train.reduced.jsonl \
--output data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv
"""
The size of collection = 8841823
Load tfidf model from: data/fb_docs/docT5query/tfidf.model.pt

Preprocess collection: converting into tfidf
100%|_________________________________________________________________________________________________________________________________| 8841823/8841823 [09:09<00:00, 16096.82it/s]
#> Load feedback documents from data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.train.reduced.jsonl (topk_docs=3)

Select expansion terms using efficient TF-IDF: fb_k=32
Elapsed time for docT5query query expansion on 502939 queries: 26.966 seconds = 0.000054 per query

Write expanded query into: data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv

qid=121352, query-expanded=define extreme [SEP] extremities extremity extreme define definition meaning of extremes is what or the ultimacy an plural example degree appendage furthest far utmost farthest most very depressed limb feelings remote urban mount apart arm
qid=634306, query-expanded=what does chattel mean on credit history [SEP] chattel slavery definition origin the chatel meaning slaves slave civil did war legal united states was history trade word end chattels of what in is usa begin chateau originate union why define
qid=920825, query-expanded=what was the great leap forward brainly [SEP] leap forward great was the what china did mao when is glf economy start party place plan why quizlet happen definition of teeping accomplish take campaign purpose rivalled chinese economic led modern
qid=510633, query-expanded=tattoo fixers how much does it cost [SEP] fixer upper cost much fixing uppers how renovation house to renovate does remodel it the zero of is should redo division up approximate price line calculate do calculator estate square foot home
qid=737889, query-expanded=what is decentralization process. [SEP] decentralization kosovo process decentralized is what the decentralisation of in kosova government governance frequently processes questions democracy why politics citizens to centralized inform komoroso legal intended respond asked decentralize role answers redefines

         output:                data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv
"""
# Elapsed time for docT5query query expansion on 502939 queries: 26.966 seconds = 0.000054 per query
# output: data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv
# 
# MSMARCO Dev: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.dev.small.jsonl
python -m preprocessing.feedback_documents.docT5query_expansion \
--collection_jsonl /workspace/DataCenter/MSMARCO/docT5query/msmarco-passage-expanded \
--tfidf_model data/fb_docs/docT5query/tfidf.model.pt \
--fb_docs 3 --fb_k 32 \
--queries data/queries.dev.small.tsv \
--fb_ranking data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.dev.small.jsonl \
--output data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k32.tsv
# Elapsed time for docT5query query expansion on 6980 queries: 0.720 seconds = 0.000103 per query
# output: data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k32.tsv
# 
# TREC-DL 2019: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2019.pass.jsonl
python -m preprocessing.feedback_documents.docT5query_expansion \
--collection_jsonl /workspace/DataCenter/MSMARCO/docT5query/msmarco-passage-expanded \
--tfidf_model data/fb_docs/docT5query/tfidf.model.pt \
--fb_docs 3 --fb_k 32 \
--queries data/queries.trec2019.tsv \
--fb_ranking data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2019.pass.jsonl \
--output data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k32.tsv
# Elapsed time for docT5query query expansion on 43 queries: 0.346 seconds = 0.008045 per query
# output: data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k32.tsv
# 
# TREC-DL 2020: data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2020.pass.jsonl
python -m preprocessing.feedback_documents.docT5query_expansion \
--collection_jsonl /workspace/DataCenter/MSMARCO/docT5query/msmarco-passage-expanded \
--tfidf_model data/fb_docs/docT5query/tfidf.model.pt \
--fb_docs 3 --fb_k 32 \
--queries data/queries.trec2020.tsv \
--fb_ranking data/fb_docs/docT5query/ranking/run.msmarco-passage-expanded.trec2020.pass.jsonl \
--output data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k32.tsv
# Elapsed time for docT5query query expansion on 54 queries: 0.347 seconds = 0.006424 per query
# output: data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k32.tsv


#* Precompute query tokenization: V1: [CLS] [Q] query with [MASK] (32 length) [SEP] expansion terms (10 length)
# MSMARCO Train: data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.v1.pt
# 
# MSMARCO Dev: data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k10.tensor.cache.v1.pt
# 
# TREC-DL 2019: data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k10.tensor.cache.v1.pt
# 
# TREC-DL 2020: data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k10.tensor.cache.v1.pt
""" Example
qid=1030303, query=who is aziz hashim
{'id': tensor([  101,     1,  2040,  2003, 21196, 23325,  5714,   102,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   102, 23325,  5714, 21196,  2040, 21196,  5313, 17212,
         2094,  2003,  6605]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'toks': ['[CLS]', '[unused0]', 'who', 'is', 'aziz', 'hash', '##im', '[SEP]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[SEP]', 'hash', '##im', 'aziz', 'who', 'aziz', '##ul', 'nr', '##d', 'is', 'managing']}

qid=1037496, query=who is rep scalise?
{'id': tensor([  101,     1,  2040,  2003, 16360,  8040, 13911,  2063,  1029,   102,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   102,  8040, 13911,  2063,  3889,  3951,  2040,  2160,
        11473,  4387,  7561]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'toks': ['[CLS]', '[unused0]', 'who', 'is', 'rep', 'sc', '##alis', '##e', '?', '[SEP]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[SEP]', 'sc', '##alis', '##e', 'steve', 'republican', 'who', 'house', 'whip', 'representative', 'error']}

qid=1043135, query=who killed nicholas ii of russia
{'id': tensor([  101,     1,  2040,  2730,  6141,  2462,  1997,  3607,   102,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   102, 17608,  6141,  2462,  2001,  2730,  3607,  2040,
         2845,  3750,  3950]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'toks': ['[CLS]', '[unused0]', 'who', 'killed', 'nicholas', 'ii', 'of', 'russia', '[SEP]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[SEP]', 'tsar', 'nicholas', 'ii', 'was', 'killed', 'russia', 'who', 'russian', 'emperor', 'buried']}

qid=1051399, query=who sings monk theme song
{'id': tensor([  101,     1,  2040, 10955,  8284,  4323,  2299,   102,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   102,  8284,  4323,  2299, 10955,  2040,  8894,  1996,
         9978,  2626,  2041]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'toks': ['[CLS]', '[unused0]', 'who', 'sings', 'monk', 'theme', 'song', '[SEP]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[SEP]', 'monk', 'theme', 'song', 'sings', 'who', 'jungle', 'the', 'monks', 'wrote', 'out']}

qid=1064670, query=why do hunters pattern their shotguns?
{'id': tensor([  101,     1,  2339,  2079,  9624,  5418,  2037, 13305,  2015,  1029,
          102,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103,   102,  9624, 13305,  2015,  5418,  2339,  2037, 13305,
         2079,  5418,  2075]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'toks': ['[CLS]', '[unused0]', 'why', 'do', 'hunters', 'pattern', 'their', 'shotgun', '##s', '?', '[SEP]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[SEP]', 'hunters', 'shotgun', '##s', 'pattern', 'why', 'their', 'shotgun', 'do', 'pattern', '##ing']}"""

#* Precompute query tokenization: V2: [CLS] [Q] query with expansion terms (32 length) [SEP] (+attention_mask on expansion terms, to prevent semantic shift)
# MSMARCO Train: data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.train.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.v2.pt
# 
# MSMARCO Dev: data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k10.tensor.cache.v2.pt
# 
# TREC-DL 2019: data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k10.tensor.cache.v2.pt
# 
# TREC-DL 2020: data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k32.tsv
python -m preprocessing.feedback_documents.precompute_expanded_qtoks --query_maxlen 32 --fb_k 10 \
--queries data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k32.tsv \
--output data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k10.tensor.cache.v2.pt
""" Example
qid=121352, query=define extreme
{'id': tensor([  101,     1,  9375,  6034,   102,  4654,  7913, 22930,  3111,  4654,
         7913, 16383,  6034,  9375,  6210,   103,   103,   103,   103,   103,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103]), 'mask': tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'exp_mask': tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'toks': ['[CLS]', '[unused0]', 'define', 'extreme', '[SEP]', 'ex', '##tre', '##mit', '##ies', 'ex', '##tre', '##mity', 'extreme', 'define', 'definition', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]']}

qid=634306, query=what does chattel mean on credit history
{'id': tensor([  101,     1,  2054,  2515, 11834,  9834,  2812,  2006,  4923,  2381,
          102, 11834,  9834,  8864,  6210,  4761,  1996, 11834,  2884,  3574,
         7179,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'toks': ['[CLS]', '[unused0]', 'what', 'does', 'chat', '##tel', 'mean', 'on', 'credit', 'history', '[SEP]', 'chat', '##tel', 'slavery', 'definition', 'origin', 'the', 'chat', '##el', 'meaning', 'slaves', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]']}

qid=920825, query=what was the great leap forward brainly
{'id': tensor([  101,     1,  2054,  2001,  1996,  2307, 11679,  2830,  4167,  2135,
          102, 11679,  2830,  2307,  2001,  1996,  2054,  2859,  2106, 15158,
         2043,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'toks': ['[CLS]', '[unused0]', 'what', 'was', 'the', 'great', 'leap', 'forward', 'brain', '##ly', '[SEP]', 'leap', 'forward', 'great', 'was', 'the', 'what', 'china', 'did', 'mao', 'when', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]']}

qid=510633, query=tattoo fixers how much does it cost
{'id': tensor([  101,     1, 11660,  8081,  2545,  2129,  2172,  2515,  2009,  3465,
          102,  8081,  2121,  3356,  3465,  2172, 15887,  3356,  2015,  2129,
        10525,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'toks': ['[CLS]', '[unused0]', 'tattoo', 'fix', '##ers', 'how', 'much', 'does', 'it', 'cost', '[SEP]', 'fix', '##er', 'upper', 'cost', 'much', 'fixing', 'upper', '##s', 'how', 'renovation', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]']}

qid=737889, query=what is decentralization process.
{'id': tensor([  101,     1,  2054,  2003, 11519,  7941,  3989,  2832,  1012,   102,
        11519,  7941,  3989, 11491,  2832, 11519,  7941,  3550,  2003,  2054,
          103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
          103,   103]), 'mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'exp_mask': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 'toks': ['[CLS]', '[unused0]', 'what', 'is', 'decent', '##ral', '##ization', 'process', '.', '[SEP]', 'decent', '##ral', '##ization', 'kosovo', 'process', 'decent', '##ral', '##ized', 'is', 'what', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]']}
"""


#* Copy results to dilab4, sonic
# copy: from dilab003 to (sonic or dilab4)
scp jihyuk@dilab003.yonsei.ac.kr:/hdd/jihyuk/Research/PassageRetrieval/qe_pseudo_labeling/data/fb_docs/docT5query/* data/fb_docs/docT5query/
# copy: from sonic to dilab4 (local connect)
scp -P 7777 sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/data/fb_docs/docT5query/* data/fb_docs/docT5query/
# added files
scp jihyuk@dilab003.yonsei.ac.kr:/hdd/jihyuk/Research/PassageRetrieval/qe_pseudo_labeling/data/fb_docs/docT5query/*.v2.pt data/fb_docs/docT5query/
scp -P 7777 sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/data/fb_docs/docT5query/*.v2.pt data/fb_docs/docT5query/



#* Copy pretrained checkpoints
# KD from ColBERT-PRF with beta=0.5: experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn
# copy: from sonic
mkdir -p experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/
scp -P 7777 sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn


#* Prepare new train triples with multiple HN
# training with small batch size, to reduce the number of queries, e.g., 18
# training with many hard negatives, to increase the number of negative documents, e.g., 8
# 
# # copy: from sonic
# mkdir -p experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/
# scp -P 7777 sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/
# 
hard_negatives=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/ranking.jsonl
n_negatives=8
sh scripts/label/msmarco_psg.triples.hn.sh ${hard_negatives} ${n_negatives}
# output: experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.hn8.jsonl
# du -hs experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.hn8.jsonl
# 3.0G	experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.hn8.jsonl
# copy: from sonic
mkdir -p experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/
scp -P 7777 sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.hn8.jsonl experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/

#* Training (fine-tuning only query encoder)
#?@ debugging
devices=0 # e.g., "0,1"
master_port=29500 # e.g., "29500"
exp_root=experiments/finetuning_query_encoder/debugging # e.g., "finetuned.b36.lr3e6.hn.kd.t1.prf.beta0.5.t2.rf.beta0.5"
cached_queries=data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.pt # e.g., "data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.pt"
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt # e.g., "experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt"
sh scripts/finetuning_query_encoder/msmarco_psg.training.query_encoder.debugging.sh ${devices} ${master_port} ${exp_root} ${cached_queries} ${kd_expansion_pt1}

#* bsize 18, hn 8
devices=0,1
master_port=29500
exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0
cached_queries=data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.v2.pt
checkpoint=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn
triples=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.hn8.jsonl
bsize=18
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt
sh scripts/finetuning_query_encoder/msmarco_psg.training.query_encoder.sh ${devices} ${master_port} ${exp_root} ${cached_queries} ${checkpoint} ${triples} ${bsize} ${kd_expansion_pt1}
clear;cat experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/train.log;echo
clear;tail -50 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/train.log;echo
# 
#* bsize 18, hn 4
devices=2,3
master_port=29600
exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0
cached_queries=data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.v2.pt
checkpoint=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn
triples=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn4.jsonl
bsize=18
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt
sh scripts/finetuning_query_encoder/msmarco_psg.training.query_encoder.sh ${devices} ${master_port} ${exp_root} ${cached_queries} ${checkpoint} ${triples} ${bsize} ${kd_expansion_pt1}
clear;cat experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/train.log;echo
clear;tail -50 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/train.log;echo
# 
#* bsize 36, hn 1
#TODO: gpu01-qexp-b36-hn1 (sonic)
devices=0,1
master_port=29500
exp_root=experiments/finetuning_query_encoder/finetuned.b36.lr3e6.hn1.kd.t1.prf.beta1.0
cached_queries=data/fb_docs/docT5query/queries.train.expanded.docs3.k10.tensor.cache.v2.pt
checkpoint=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-150000.dnn
triples=experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl
bsize=36
kd_expansion_pt1=experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta1.0.clusters24/label.py/2021-10-24_01.23.04/expansion.pt
sh scripts/finetuning_query_encoder/msmarco_psg.training.query_encoder.sh ${devices} ${master_port} ${exp_root} ${cached_queries} ${checkpoint} ${triples} ${bsize} ${kd_expansion_pt1}
clear;cat experiments/finetuning_query_encoder/finetuned.b36.lr3e6.hn1.kd.t1.prf.beta1.0/train.log;echo
clear;tail -50 experiments/finetuning_query_encoder/finetuned.b36.lr3e6.hn1.kd.t1.prf.beta1.0/train.log;echo

# indexing
indexing_devices=2,3,6,7
faiss_devices=6,7
exp_root=experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf
step=150000
sh scripts/indexing/indexing.sh ${indexing_devices} ${faiss_devices} ${exp_root} ${step}


#* retrieve & rerank (skip validation, as we have fixed index)
#* MSMARCO Dev
# exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0
exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0
# exp_root=experiments/finetuning_query_encoder/finetuned.b36.lr3e6.hn1.kd.t1.prf.beta1.0
sh scripts/finetuning_query_encoder/msmarco_psg.ranking.sh \
data/fb_docs/docT5query/queries.dev.small.expanded.docs3.k10.tensor.cache.v2.pt \
experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/index.py \
${exp_root} 150000 4
# 
#* TREC-DL 2019
# exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0
exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0
# exp_root=experiments/finetuning_query_encoder/finetuned.b36.lr3e6.hn1.kd.t1.prf.beta1.0
sh scripts/finetuning_query_encoder/trec2019_psg.ranking.sh \
data/fb_docs/docT5query/queries.trec2019.expanded.docs3.k10.tensor.cache.v2.pt \
experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/index.py \
${exp_root} 150000 4
# 
#* TREC-DL 2020
# exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0
exp_root=experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0
# exp_root=experiments/finetuning_query_encoder/finetuned.b36.lr3e6.hn1.kd.t1.prf.beta1.0
sh scripts/finetuning_query_encoder/trec2020_psg.ranking.sh \
data/fb_docs/docT5query/queries.trec2020.expanded.docs3.k10.tensor.cache.v2.pt \
experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/index.py \
${exp_root} 150000 4
# 
#?@ debugging
# rm -r experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/MSMARCO-psg/retrieve.py 
# rm -r experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/MSMARCO-psg/rerank.py 

#* Results
#* bsize 18, hn 8
clear
echo;tail -7 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/colbert-50000.dnn/MSMARCO-psg/e2e.metrics;echo
echo;tail -7 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/colbert-100000.dnn/MSMARCO-psg/e2e.metrics;echo
# 
clear
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/colbert-50000.dnn/TREC2019-psg/e2e.metrics;echo
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn8.kd.t1.prf.beta1.0/colbert-100000.dnn/TREC2019-psg/e2e.metrics;echo
#* bsize 18, hn 4
clear
echo;tail -7 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-50000.dnn/MSMARCO-psg/e2e.metrics;echo
echo;tail -7 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-100000.dnn/MSMARCO-psg/e2e.metrics;echo
echo;tail -7 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-150000.dnn/MSMARCO-psg/e2e.metrics;echo
# 
clear
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-50000.dnn/TREC2019-psg/e2e.metrics;echo
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-100000.dnn/TREC2019-psg/e2e.metrics;echo
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-150000.dnn/TREC2019-psg/e2e.metrics;echo
# 
clear
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-50000.dnn/TREC2020-psg/e2e.metrics;echo
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-100000.dnn/TREC2020-psg/e2e.metrics;echo
echo;tail -1 experiments/finetuning_query_encoder/finetuned.b18.lr3e6.hn4.kd.t1.prf.beta1.0/colbert-150000.dnn/TREC2020-psg/e2e.metrics;echo

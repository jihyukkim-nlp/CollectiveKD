import argparse
import os
from collections import OrderedDict
import torch
import ujson
import re
import numpy as np
from tqdm import tqdm 
from collections import Counter
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer

def load_collection_jsonl(path):
    line_idx = 0
    collection = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            for line in f:
                if line_idx % (1000*1000) == 0:
                    print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

                doc = ujson.loads(line)
                
                content = doc["contents"]
                # content = re.sub(r'\[SEP\] ', '', content)
                content = content.replace("[SEP] ", "")
                
                docid = int(doc['id'])

                collection[docid] = content.strip().lower()
                
                # assert int(docid) == line_idx, f'docid={docid}, line_idx={line_idx}'

                line_idx += 1

    return collection

def load_feedback_documents(path, topk_docs):
    print(f'#> Load feedback documents from {path} (topk_docs={topk_docs})')
    fb_docs = {}
    with open(path) as f:
        for line_idx, line in enumerate(f):
            qid, topk_pids = line.strip().split('\t')
            qid = int(qid)
            topk_pids = ujson.loads(topk_pids)
            topk_pids = topk_pids[:topk_docs]
            topk_pids = [_[0] for _ in topk_pids]
            fb_docs[qid] = topk_pids
    return fb_docs

def load_queries(path):
    print(f"#> Loading queries... from: {path}")

    queries = OrderedDict()

    with open(path) as f:
        for line in f:
            qid, query = line.strip().split('\t')
            qid = int(qid)
            queries[qid] = query

    return queries

def preprocess_collection(collection, tfidf_model):
    
    print(f'\n\nPreprocess collection: converting into tfidf')
    
    collection_tfidf = {} # Dict[int:Dict[int:float]]
    
    idf = tfidf_model.idf_
    w2i = tfidf_model.vocabulary_
    i2w = {i:w for w, i in w2i.items()}
    
    for docid, content in tqdm(collection.items()):
        word_ids = [w2i[w] for w in content.strip().split() if w in w2i]
        word_tf = Counter(word_ids)
        word_tfidf = {id:tf*idf[id] for id, tf in word_tf.items()}
        id_score = list(word_tfidf.items())

        # tfidf_score = tfidf_model.transform([content])
        # nonzero_idxs = tfidf_score.nonzero()[1]
        # nonzero_scores = tfidf_score[:, nonzero_idxs].toarray()[0, :]
        # id_score = list(zip(nonzero_idxs, nonzero_scores))
        # id_score: List [ Tuple ( int, float) ]

        collection_tfidf[docid] = dict(id_score)
    
    return collection_tfidf

def merge_tfidf(org, add):
    # org, add: Dict[int:float]
    for _id, _tfidf in add.items():
        org[_id] = org.get(_id, 0.0)
        org[_id] = org[_id] + _tfidf
    return org

def selection_expansion_term(queries, fb_docs, collection, collection_tfidf, tfidf_model, fb_k):
    print(f'\n\nSelect expansion terms using efficient TF-IDF: fb_k={fb_k}')
    
    w2i = tfidf_model.vocabulary_
    i2w = {i:w for w, i in w2i.items()}

    queries_expaned = OrderedDict()
    for qid, query in queries.items():
        if qid in fb_docs:
            pids = fb_docs[qid]
            # print(f'\n\n\nqid={qid}, pids={pids}')
            
            id_score = {} # Dict[int:float]
            for pid in pids:
                id_score = merge_tfidf(id_score, collection_tfidf[pid])
            id_score = id_score.items()

            sorted_id_score = sorted(id_score, key=lambda x:-x[1])
            topk_id_score = sorted_id_score[:fb_k]
            topk_wordids = [_[0] for _ in topk_id_score]
            
            expansion_terms = [i2w[idx] for idx in topk_wordids]
            # print(f'\nexpansion_terms={expansion_terms}')
            
            query_expanded = query + ' [SEP] ' + " ".join(expansion_terms)
            queries_expaned[qid] = query_expanded
        else:
            query_expanded = query + ' [SEP]'
            queries_expaned[qid] = query_expanded
    
    return queries_expaned

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--queries', help="path/to/queries.tsv")
    parser.add_argument('--collection_jsonl', help="path/to/msmarco-passage-expanded, that containing doc00.json, doc01.json, ...")
    parser.add_argument('--tfidf_model', help="path/to/tfidf.model.pt")
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_features', type=int, default=None)
    parser.add_argument('--fb_ranking', dest='fb_ranking', help="/path/to/label.py/*/ranking.jsonl, containing feedback documents for each query")
    parser.add_argument('--fb_docs', dest='fb_docs', default=3, type=int, help="The # of feedback documents used for query expansion.")
    parser.add_argument('--fb_k', dest='fb_k', default=10, type=int, help="The # of expansion terms used for query expansion.")
    parser.add_argument('--output', required=True)
    

    args = parser.parse_args()

    queries = load_queries(args.queries)
    qids_in_order = list(queries.keys())

    collection = load_collection_jsonl(path=args.collection_jsonl)
    print(f'The size of collection = {len(collection)}')

    if not os.path.exists(args.tfidf_model):
        """
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?',
            'The last document?',
        ]
        """
        corpus = [collection[pid] for pid in collection]
        tfidf_model = TfidfVectorizer(min_df=args.min_df, max_features=args.max_features).fit(corpus)
        torch.save(tfidf_model, args.tfidf_model)
        print(f'Save tfidf model into: {args.tfidf_model}')
    else:
        print(f'Load tfidf model from: {args.tfidf_model}')
        tfidf_model = torch.load(args.tfidf_model)
        
    # col_tfidf_cache = args.tfidf_model+'.col_cache'
    # if os.path.exists(col_tfidf_cache):
    #     print(f'Load tfidf for collection from: {col_tfidf_cache}')
    #     collection_tfidf = torch.load(col_tfidf_cache)
    # else:
    #     collection_tfidf = preprocess_collection(collection=collection, tfidf_model=tfidf_model)
    #     print(f'Save tfidf for collection into: {col_tfidf_cache}')
    #     torch.save(collection_tfidf, col_tfidf_cache)
    collection_tfidf = preprocess_collection(collection=collection, tfidf_model=tfidf_model)

    fb_docs = load_feedback_documents(path=args.fb_ranking, topk_docs=args.fb_docs)

    st = time()
    # Select expansion terms using efficient TF-IDF: fb_k=10
    queries_expaned = selection_expansion_term(queries=queries, fb_docs=fb_docs, collection=collection, collection_tfidf=collection_tfidf, tfidf_model=tfidf_model, fb_k=args.fb_k)
    ed = time()
    print(f'Elapsed time for docT5query query expansion on {len(queries_expaned)} queries: {ed-st:.3f} seconds = {(ed-st)/len(queries_expaned):.6f} per query')
    # Elapsed time for query expansion from docT5query: 24.723 seconds


    print(f'\n\nWrite expanded query into: {args.output}\n')
    with open(args.output, 'w', encoding='utf-8') as outfile:
        for query_idx, (qid, query) in enumerate(queries_expaned.items()):
            if query_idx < 5:
                print(f'qid={qid}, query-expanded={query}')
            outfile.write(f'{qid}\t{query}\n')
    print(f'\n\t output: \t\t{args.output}\n')
    
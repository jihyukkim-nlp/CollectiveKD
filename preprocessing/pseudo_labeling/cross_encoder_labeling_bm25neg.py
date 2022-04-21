import argparse
import os
import json
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import torch
import random

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoBERT, MonoT5

from time import time

def load_queries(queries_path):
    queries = OrderedDict()

    print("#> Loading the queries from", queries_path, "...")

    with open(queries_path) as f:
        for line in f:
            qid, query, *_ = line.strip().split('\t')
            qid = int(qid)

            assert (qid not in queries), ("Query QID", qid, "is repeated!")
            queries[qid] = query

    print("#> Got", len(queries), "queries. All QIDs are unique.\n")

    return queries

def load_collection(path):
    print("#> Loading collection...")

    collection = []

    with open(path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % (1000*1000) == 0:
                print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

            pid, passage = line.strip().split('\t')
            assert int(pid) == line_idx
            collection.append(passage)
    print()

    return collection

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reranker', choices=['MonoBERT', 'MonoT5', 'MonoT5Large'], default='MonoBERT')
    parser.add_argument('--queries', dest='queries', required=True)
    parser.add_argument('--triples', dest='triples', required=True)
    parser.add_argument('--collection', dest='collection', required=True)
    # parser.add_argument('--output', required=True)

    args = parser.parse_args()

    queries = load_queries(args.queries)
    collection = load_collection(args.collection)

    assert os.path.exists(args.triples)

    print(f'\n#> Load {args.reranker}')
    # reranker_dict = {
    #     'MonoBERT': MonoBERT,
    #     'MonoT5': MonoT5,
    # }
    # reranker =  reranker_dict[args.reranker]()
    if args.reranker == 'MonoBERT':
        reranker = MonoBERT()
    elif args.reranker == 'MonoT5':
        reranker = MonoT5()
    elif args.reranker == 'MonoT5Large':
        reranker = MonoT5(pretrained_model_name_or_path='castorini/monot5-large-msmarco')
    print(f'#> [Done] {args.reranker} is loaded')

    print(f'\n#> Load {args.triples}')
    qid_to_pids = defaultdict(set)
    with torch.no_grad():
        # n_lines = sum(1 for _ in open(args.triples))
        with open(args.triples) as ifile:
            # for line_idx, line in enumerate(tqdm(ifile, total=n_lines)):
            for line_idx, line in enumerate(ifile):
                qid, *pids = json.loads(line)

                qid_to_pids[qid].update(pids)

                # if line_idx==100: break
    
    print(f'#> [Done] {args.triples} is loaded')
    # print(f'#> The # of queries: {len(qid_to_pids)}')
    n_all_queries = 502939 # awk -F '\t' '{ print $1 }' data/qrels.train.tsv | sort -n | uniq | wc -l 
    print(f'#> The # of queries: {n_all_queries}')

    n_samples = 1000
    print(f'\n#> Sample {n_samples} queries as an approximation')
    sampled_queries = list(qid_to_pids.keys())
    random.shuffle(sampled_queries)
    sampled_queries = sampled_queries[:n_samples]
    sampled_queries = set(sampled_queries)
    qid_to_pids = {qid:pids for qid, pids in qid_to_pids.items() if qid in sampled_queries}
    assert len(qid_to_pids) == n_samples

    st = time()
    print(f'\n#> Start re-ranking!')
    with torch.no_grad():
        for qid, pids in tqdm(qid_to_pids.items()):
            query = queries[qid]
            query = Query(query)
            texts = [Text(collection[pid], {'docid':str(pid)}, 0) for pid in pids]
            reranked = reranker.rerank(query, texts)

            # for rank, r in enumerate(reranked):
            #     pid = r.metadata["docid"]
            #     score = r.score
            #     outfile.write(f'{qid}\t{pid}\t{rank+1}\t{score}\n')

    ed = time()
    elapsed = ed-st
    print(f'Time consumption: {elapsed} seconds for {n_samples}')
    print(f'Approximated time for {n_all_queries} queries: {elapsed * (n_all_queries / n_samples):.3f} seconds = {elapsed * (n_all_queries / n_samples) / 3600:.3f} hours')




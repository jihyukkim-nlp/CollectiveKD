import ujson
import csv
import os
import re
from tqdm import tqdm

from colbert.evaluation.loaders import load_queries

import argparse


from collections import defaultdict, OrderedDict
def load_qrels(qrels_path):
    if qrels_path is None:
        return None

    print("#> Loading qrels from", qrels_path, "...")

    #!@ custom
    qrels = OrderedDict()
    with open(qrels_path, mode='r', encoding="utf-8") as f:
        for line in f:
            qid, _, pid, _ = line.strip().split()
            qid, pid = map(int, (qid, pid))
            qrels[qid] = qrels.get(qid, set())
            qrels[qid].add(pid)
    for qid in qrels:
        qrels[qid] = list(qrels[qid])

    assert all(len(qrels[qid]) == len(set(qrels[qid])) for qid in qrels)

    avg_positive = round(sum(len(qrels[qid]) for qid in qrels) / len(qrels), 2)

    print("#> Loaded qrels for", len(qrels), "unique queries with",
                  avg_positive, "positives per query on average.\n")

    return qrels



if __name__=='__main__':
    
    # DATA_DIR = '/workspace/DataCenter/PassageRanking/MSMARCO' # sonic
    # DATA_DIR = '/workspace/DataCenter/MSMARCO' # dilab003
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', dest='queries', default='/workspace/DataCenter/MSMARCO/queries.train.tsv')
    parser.add_argument('--qrels', dest='qrels', default='/workspace/DataCenter/MSMARCO/qrels.train.tsv')
    parser.add_argument('--out', dest='out', default='data/queries.train.reduced.tsv')

    args = parser.parse_args()

    print(f'What to do?\n\t ===> Reduce ``{args.queries}``, by only retaining queries that are included in ``{args.qrels}``.')

    def reduce_queries(queries, qrels):
        qids_in_order = list(queries.keys()).copy()
        for qid in qids_in_order:
            if qid not in qrels:
                del queries[qid]
        print(f'#> Reduce the # of queries: {len(qids_in_order)} -> {len(queries)}')


    qrels = load_qrels(args.qrels)
    print(f'#> The # of queries in qrels = {len(qrels)}')

    queries = load_queries(args.queries)
    reduce_queries(queries, qrels)

    outfile_path = args.out
    print(f'#> Write reduced queries into "{outfile_path}"')
    with open(outfile_path, 'w', encoding='utf-8') as outfile:
        for qid, query in queries.items():
            outfile.write(f'{qid}\t{query}\n')
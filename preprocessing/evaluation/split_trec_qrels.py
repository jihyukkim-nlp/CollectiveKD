import argparse
from tqdm import tqdm
import numpy as np

from collections import defaultdict, OrderedDict

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

def load_qrels(qrels_path):
    if qrels_path is None:
        return None

    print("#> Loading qrels from", qrels_path, "...")

    #!@ custom
    qrels = OrderedDict()
    with open(qrels_path, mode='r', encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            qid, _, pid, rel = line.strip().split()
            if line_idx < 5:
                print(f'({line_idx})\t{line.strip()}')
            qid, pid, rel = map(int, (qid, pid, rel))
            qrels[qid] = qrels.get(qid, set())
            qrels[qid].add((pid, rel))
    for qid in qrels:
        qrels[qid] = list(qrels[qid])

    assert all(len(qrels[qid]) == len(set(qrels[qid])) for qid in qrels)

    avg_positive = round(sum(len([_ for _ in qrels[qid] if _[1]>0]) for qid in qrels) / len(qrels), 2)

    print("#> Loaded qrels for", len(qrels), "unique queries with",
                  avg_positive, "positives per query on average.\n")

    return qrels





if __name__=='__main__':
    
    # DATA_DIR = '/workspace/DataCenter/PassageRanking/MSMARCO' # sonic
    # DATA_DIR = '/workspace/DataCenter/MSMARCO' # dilab003
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrels', dest='qrels', default='/workspace/DataCenter/MSMARCO/trec_evaluation/2019qrels-pass.tsv')

    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    print(f'#> The # of queries in qrels = {len(qrels)}')

    qids_in_order = list(qrels.keys())

    tag = 2019 if '2019' in args.qrels else 2020
    args.outfile_train = f'data/{tag}qrels-pass.train.tsv' 
    args.outfile_test = f'data/{tag}qrels-pass.test.tsv' 
    print(f'outfile_train={args.outfile_train}')
    print(f'outfile_test={args.outfile_test}')

    outfile_train = open(args.outfile_train, 'w')
    outfile_test = open(args.outfile_test, 'w')

    for qid in qids_in_order:
        # pid_rels = qrels[qid] # List[Tuple(int, int)]
        pid_rels = sorted(qrels[qid], key=lambda x: -x[1])
        
        _train = pid_rels[0]
        _test = pid_rels[1:]
        assert _train[1] > 1


        if tag==2019:
            # 19335 Q0 1017759 0
            pid, rel = _train
            outfile_train.write(f'{qid} Q0 {pid} {rel}\n')

            for pid, rel in _test:
                outfile_test.write(f'{qid} Q0 {pid} {rel}\n')

        elif tag==2020:
            # 23849 0 1020327 2
            pid, rel = _train
            outfile_train.write(f'{qid} 0 {pid} {rel}\n')

            for pid, rel in _test:
                outfile_test.write(f'{qid} 0 {pid} {rel}\n')
                
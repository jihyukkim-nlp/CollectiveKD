import csv
import argparse
import collections
import numpy as np

def load_qids(path):
    qid_set = set()
    with open(path) as ifile:
        for line in ifile:
            qid = line.strip().split('\t')[0]
            qid_set.add(qid)
    return qid_set

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', default='data/queries.train.reduced.tsv')
    parser.add_argument('--ranking', default='./experiments/pqa_colbert-s20-b36-lr3e6/MSMARCO-psg-pruned-s20-train-fb_embs5-beta1.0/prf.py/queries.train.reduced.prf_topK_pids.jsonl')
    
    args = parser.parse_args()

    qid_set1 = load_qids(args.queries)
    qid_set2 = load_qids(args.ranking)

    print(f'#> The # of queries in "{args.queries}" = {len(qid_set1)}')
    print(f'#> The # of ranking in "{args.ranking}" = {len(qid_set2)}')

    assert not (qid_set1 - qid_set2)
    assert not (qid_set2 - qid_set1)
    
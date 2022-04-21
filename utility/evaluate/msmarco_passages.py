"""
    Evaluate MS MARCO Passages ranking.
"""

import os
import math
import tqdm
import ujson
import random

import pandas as pd
import numpy as np
import pyterrier as pt

from argparse import ArgumentParser
from collections import defaultdict
from colbert.utils.utils import print_message, file_tqdm

def load_qrels(qrels_path):
    if qrels_path is None:
        return None

    print("#> Loading qrels from", qrels_path, "...")

    qid_list = []
    pid_list = []
    rel_list = []
    with open(qrels_path, mode='r', encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            qid, _, pid, rel = line.strip().split()
            qid, pid, rel = map(int, (qid, pid, rel))
            qid_list.append(str(qid))
            pid_list.append(str(pid))
            rel_list.append(rel)
    qrels = pd.DataFrame({
        'qid':qid_list,
        'docno':pid_list,
        'label':np.array(rel_list, dtype=np.int64),
    })
    return qrels

def load_ranking(path, qrels_exclude=None):
    print("#> Loading ranking from", path, "...")

    qid_list = []
    pid_list = []
    rank_list = []
    score_list = []

    with open(path, mode='r', encoding="utf-8") as f:
        
        if path.endswith('.jsonl'):
            for line_idx, line in enumerate(f):
                qid, pids = line.strip().split('\t')
                pids = ujson.loads(pids)
                _rank = 1
                for rank, pid in enumerate(pids):
                    pid = str(pid)
                    if (qrels_exclude is None) or pid not in qrels_exclude[qid]:
                        qid_list.append(qid)
                        pid_list.append(pid)
                        rank_list.append(_rank)
                        score_list.append(1000-float(_rank))
                        _rank += 1

        elif path.endswith('.tsv'):
            qid_rank = defaultdict(int)
            for line_idx, line in enumerate(f):
                qid, pid, rank, score = line.strip().split('\t')
                if (qrels_exclude is None) or pid not in qrels_exclude[qid]:
                    qid_rank[qid] += 1
                    _rank = qid_rank[qid]

                    qid_list.append(qid)
                    pid_list.append(pid)
                    rank_list.append(_rank)
                    score_list.append(1000-float(_rank))

    ranking = pd.DataFrame({
        'qid':qid_list,
        'docno':pid_list,
        'rank':np.array(rank_list, dtype=np.int64),
        'score':np.array(score_list, dtype=np.float64),
    })
    return ranking

def main(args):
    qrels_trec = load_qrels(args.qrels)
    ranking_trec = load_ranking(args.ranking)
    if not pt.started(): pt.init()
    from pyterrier.measures import RR, nDCG, AP, NumRet, R, P
    eval = pt.Utils.evaluate(ranking_trec, qrels_trec, 
        metrics=[
            RR(rel=1)@10, nDCG@10, R(rel=1)@1000, AP(rel=1)@1000, 
            NumRet, "num_q",
        ],
        # These measures are from "https://github.com/terrierteam/ir_measures/tree/f6b5dc62fd80f9e4ca5678e7fc82f6e8173a800d/ir_measures/measures"
    )
    # print(eval)
    
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)
    qid2mrr = {}
    qid2mrr100 = {}
    qid2recall = {depth: {} for depth in [50, 200, 1000]}

    with open(args.qrels) as f:
        print_message(f"#> Loading QRELs from {args.qrels} ..")
        for line in file_tqdm(f):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1

            qid2positives[qid].append(pid)
            
    with open(args.ranking) as f:
        print_message(f"#> Loading ranked lists from {args.ranking} ..")
        for line in file_tqdm(f):
            qid, pid, rank, *score = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)

            if len(score) > 0:
                assert len(score) == 1
                score = float(score[0])
            else:
                score = None

            qid2ranking[qid].append((rank, pid, score))

    assert set.issubset(set(qid2ranking.keys()), set(qid2positives.keys()))

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)

    if num_judged_queries != num_ranked_queries:
        print()
        print_message("#> [WARNING] num_judged_queries != num_ranked_queries")
        print_message(f"#> {num_judged_queries} != {num_ranked_queries}")
        print()

    print_message(f"#> Computing MRR@10 for {num_judged_queries} queries.")

    for qid in tqdm.tqdm(qid2positives):
        ranking = qid2ranking[qid]
        positives = qid2positives[qid]

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                if rank <= 10:
                    qid2mrr[qid] = 1.0 / rank
                break
        
        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                if rank <= 100:
                    qid2mrr100[qid] = 1.0 / rank
                break

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                for depth in qid2recall:
                    if rank <= depth:
                        qid2recall[depth][qid] = qid2recall[depth].get(qid, 0) + 1.0 / len(positives)

    assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)

    print()

    mrr_10_sum = sum(qid2mrr.values())
    print_message(f"#> MRR@10 = {mrr_10_sum / num_judged_queries}")
    # print_message(f"#> MRR@10 (only for ranked queries) = {mrr_10_sum / num_ranked_queries}")
    # print()

    mrr_100_sum = sum(qid2mrr100.values())
    print_message(f"#> MRR@100 = {mrr_100_sum / num_judged_queries}")
    
    for depth in qid2recall:
        assert len(qid2recall[depth]) <= num_ranked_queries, (len(qid2recall[depth]), num_ranked_queries)

        # print()
        metric_sum = sum(qid2recall[depth].values())
        print_message(f"#> Recall@{depth} = {metric_sum / num_judged_queries}")
        # print_message(f"#> Recall@{depth} (only for ranked queries) = {metric_sum / num_ranked_queries}")
        # print()

    if args.annotate:
        print_message(f"#> Writing annotations to {args.output} ..")

        with open(args.output, 'w') as f:
            for qid in tqdm.tqdm(qid2positives):
                ranking = qid2ranking[qid]
                positives = qid2positives[qid]

                for rank, (_, pid, score) in enumerate(ranking):
                    rank = rank + 1  # 1-indexed
                    label = int(pid in positives)

                    line = [qid, pid, rank, score, label]
                    line = [x for x in line if x is not None]
                    line = '\t'.join(map(str, line)) + '\n'
                    f.write(line)

    ndcg_at_10 = eval["nDCG@10"]
    map_at_1000 = eval["AP@1000"]
    print_message(f"#> NDCG@10 = {ndcg_at_10}")
    print_message(f"#> MAP@1000 = {map_at_1000}")

if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--annotate', dest='annotate', default=False, action='store_true')

    args = parser.parse_args()

    if args.annotate:
        args.output = f'{args.ranking}.annotated'
        assert not os.path.exists(args.output), args.output

    main(args)

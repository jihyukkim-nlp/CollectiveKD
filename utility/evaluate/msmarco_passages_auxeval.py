"""
    Evaluate MS MARCO Passages ranking.
"""

import os
import math
import tqdm
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict
from colbert.utils.utils import print_message, file_tqdm


def main(args):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)

    MAX_DEPTH=10
    qid2precision = {depth: {} for depth in [1, 3, 5, 10]}
    qid2recall = {depth: {} for depth in [1, 3, 5, 10]}
    qid2hit = {depth: {} for depth in [1, 3, 5, 10]}
    qid2mrr = {}
    # qid2mrr100 = {}
    
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

            if qid not in qid2positives:
                continue

            if rank > MAX_DEPTH:
                continue

            if len(score) > 0:
                assert len(score) == 1
                score = float(score[0])
            else:
                score = None

            qid2ranking[qid].append((rank, pid, score))

    assert set.issubset(set(qid2ranking.keys()), set(qid2positives.keys()))

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)
    assert num_judged_queries == num_ranked_queries
    # if num_judged_queries != num_ranked_queries:
    #     print()
    #     print_message("#> [WARNING] num_judged_queries != num_ranked_queries")
    #     print_message(f"#> {num_judged_queries} != {num_ranked_queries}")
    #     print()

    print_message(f"#> Computing MRR@10/100 & Precision/Recall for {num_judged_queries} queries.")

    for qid in tqdm.tqdm(qid2positives):
        ranking = qid2ranking[qid]
        positives = qid2positives[qid]

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                if rank <= 10:
                    qid2mrr[qid] = 1.0 / rank
                break
        
        # for rank, (_, pid, _) in enumerate(ranking):
        #     rank = rank + 1  # 1-indexed

        #     if pid in positives:
        #         if rank <= 100:
        #             qid2mrr100[qid] = 1.0 / rank
        #         break

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                
                for depth in qid2recall:
                    if rank <= depth:
                        qid2recall[depth][qid] = qid2recall[depth].get(qid, 0) + 1.0 / len(positives)
                
                for depth in qid2precision:
                    if rank <= depth:
                        qid2precision[depth][qid] = qid2precision[depth].get(qid, 0) + 1.0 / depth

                for depth in qid2hit:
                    if rank <= depth:
                        qid2hit[depth][qid] = 1.0


    assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)

    print()

    mrr_10_sum = sum(qid2mrr.values())
    print_message(f"#> MRR@10 = {mrr_10_sum / num_judged_queries}")
    
    # mrr_100_sum = sum(qid2mrr100.values())
    # print_message(f"#> MRR@100 = {mrr_100_sum / num_judged_queries}")
    
    for depth in qid2recall:
        assert len(qid2recall[depth]) <= num_ranked_queries, (len(qid2recall[depth]), num_ranked_queries)
        metric_sum = sum(qid2recall[depth].values())
        print_message(f"#> Recall@{depth} = {metric_sum / num_judged_queries}")
    
    for depth in qid2precision:
        assert len(qid2precision[depth]) <= num_ranked_queries, (len(qid2precision[depth]), num_ranked_queries)
        metric_sum = sum(qid2precision[depth].values())
        print_message(f"#> Precision@{depth} = {metric_sum / num_judged_queries}")
    
    for depth in qid2hit:
        assert len(qid2hit[depth]) <= num_ranked_queries, (len(qid2hit[depth]), num_ranked_queries)
        metric_sum = sum(qid2hit[depth].values())
        print_message(f"#> Hit@{depth} = {metric_sum / num_judged_queries}")


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)

    args = parser.parse_args()

    main(args)

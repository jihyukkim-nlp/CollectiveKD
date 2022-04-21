"""
    Significance test on per-query ranking performance (Student's paired t-test)
"""

import os
import math
from types import prepare_class
import tqdm
import ujson
import random
import tqdm

import pandas as pd
import numpy as np

from argparse import ArgumentParser
from collections import defaultdict

from scipy.stats import ttest_ind, wilcoxon

def load_per_query_performance(path):
    mrr_10 = {}
    ndcg_10 = {}
    map_1000 = {}
    recall_1000 = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line_idx, line in enumerate(file):
            
            qid, perf_dict = line.strip().split('\t')
            perf_dict = ujson.loads(perf_dict)
            
            mrr_10[int(qid)] = perf_dict["RR@10"]
            ndcg_10[int(qid)] = perf_dict["nDCG@10"]
            
            recall_1000[int(qid)] = perf_dict["R@1000"]
            map_1000[int(qid)] = perf_dict["AP@1000"]

    return mrr_10, ndcg_10, map_1000, recall_1000


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--perf_a', dest='perf_a', required=True, type=str, help="Performance of proposed model.")
    parser.add_argument('--perf_b', dest='perf_b', required=True, type=str, help="Performance of baseline model.")
    
    args = parser.parse_args()

    assert os.path.exists(args.perf_a)
    assert os.path.exists(args.perf_b)

    perf_a, perf_b = {}, {}
    perf_a["mrr_10"], perf_a["ndcg_10"], perf_a["map_1000"], perf_a["recall_1000"] = load_per_query_performance(args.perf_a)
    perf_b["mrr_10"], perf_b["ndcg_10"], perf_b["map_1000"], perf_b["recall_1000"] = load_per_query_performance(args.perf_b)

    qids = set(list(perf_a["mrr_10"].keys()))
    assert qids == set(list(perf_b["mrr_10"].keys())), f'A ({len(qids)} unique queries) and B ({len(set(list(perf_b["mrr_10"].keys())))} unique queries) has different queries.'
    qids = list(qids)
    print(f'#> The number of unique queries = {len(qids)}')

    for metric in ["mrr_10", "ndcg_10", "map_1000", "recall_1000"]:
        a = np.array([perf_a[metric][qid] for qid in qids])
        b = np.array([perf_b[metric][qid] for qid in qids])

        """
        Reference
        CIKM 2007. "A Comparison of Statistical Significance Tests for Information Retrieval Evaluation"
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.134.5558&rep=rep1&type=pdf
        """

        #* Permutation test (or randomization test)
        """
        Null Hypothesis: 
            Suppose system N generates both results from system A and results from system B.
            Under the null hypothesis,
                given L samples, 
                any permutation of labeling each result with either A or B, 
                i.e., any of 2^L permutations where 2 indicates two choices between A or B,
                is an equally likely output.
        
        Codes:
            from scipy.stats import ttest_ind
            a = np.array(performance of the proposed system for all queries)
            b = np.array(performance of the baseline system for all queries)
            statistic, pvalue = ttest_ind(a=, b=b, permutations=100000, nan_policy='raise', alternative='greater').pvalue
        """
        # pvalue = ttest_ind(a=a, b=b, permutations=100000, nan_policy='raise', alternative='greater').pvalue
        # print(f'[{metric:12s}] Proposed ({a.mean():.3f}) > Baseline ({b.mean():.3f}) at p < {pvalue:.3f}')


        #* t-test (or Student's paired t-test)
        """ 
        Codes:
            from scipy.stats import ttest_ind
            a = np.array(performance of the proposed system for all queries)
            b = np.array(performance of the baseline system for all queries)
            statistic, pvalue = ttest_ind(a=a, b=b, nan_policy='raise', alternative='greater')
        """
        pvalue = ttest_ind(a=a, b=b, nan_policy='raise', alternative='greater').pvalue
        print(f'[{metric:12s}] Proposed ({a.mean():.3f}) > Baseline ({b.mean():.3f}) at p < {pvalue:.3f}')

        #* Bootstrap test
        """
        # Compute mean of all forces: mean_force
        mean_force = np.mean(forces_concat)

        # Generate shifted arrays
        force_a_shifted = force_a - np.mean(force_a) + mean_force
        force_b_shifted = force_b - np.mean(force_b) + mean_force

        # Compute 10,000 bootstrap replicates from shifted arrays
        bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
        bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

        # Get replicates of difference of means: bs_replicates
        bs_replicates = bs_replicates_a - bs_replicates_b

        # Compute and print p-value: p
        p = np.sum(bs_replicates >= empirical_diff_means) / len(bs_replicates)
        print('p-value =', p)
        """
        
        #* Wilcoxon Singned-Ranked Test
        """
        Codes:
            from scipy.stats import wilcoxon
            a = np.array(performance of system A for all queries)
            b = np.array(performance of system B for all queries)
            diff = a - b
            w, pvalue = wilcoxon(diff)
        """
        # d = a - b
        # if np.median(d) > 0:
        #     w, pvalue = wilcoxon(d, alternative='greater')
        #     print(f'[{metric:12s}] A ({a.mean():.3f}) > B ({b.mean():.3f}) at {pvalue:.3f}')
        # else:
        #     w, pvalue = wilcoxon(-d, alternative='greater')
        #     print(f'[{metric:12s}] B ({b.mean():.3f}) > A ({a.mean():.3f}) at {pvalue:.3f}')

import argparse
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import ujson
import pandas as pd

from collections import defaultdict, OrderedDict

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

def load_ranking(path, qrels_exclude):
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
                    if pid not in qrels_exclude[qid]:
                        qid_list.append(qid)
                        pid_list.append(pid)
                        rank_list.append(_rank)
                        score_list.append(1000-float(_rank))
                        _rank += 1

        elif path.endswith('.tsv'):
            qid_rank = defaultdict(int)
            for line_idx, line in enumerate(f):
                qid, pid, rank, score = line.strip().split('\t')
                if pid not in qrels_exclude[qid]:
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



import pyterrier as pt

if __name__=='__main__':
    
    # DATA_DIR = '/workspace/DataCenter/PassageRanking/MSMARCO' # sonic
    # DATA_DIR = '/workspace/DataCenter/MSMARCO' # dilab003
    """
    print('#> Pyterrier sanity check: start: '+'#'*30)
    if not pt.started():
        pt.init()
    
    vaswani_dataset = pt.datasets.get_dataset("vaswani")
    indexref = vaswani_dataset.get_index()
    index = pt.IndexFactory.of(indexref)

    print(index.getCollectionStatistics().toString())

    topics = vaswani_dataset.get_topics()
    topics.head(5)

    retr = pt.BatchRetrieve(index, controls = {"wmodel": "TF_IDF"})

    retr.setControl("wmodel", "TF_IDF")
    retr.setControls({"wmodel": "TF_IDF"})

    res=retr.transform(topics)

    ranking = res.copy()
    del ranking['docid']
    del ranking['query']
    print('\nranking ===> ')
    print(ranking.head())
    print(f'dtypes: ranking.iloc[0] => {[type(_) for _ in ranking.iloc[0]]}')
    print(f'dtypes: ranking.iloc[1] => {[type(_) for _ in ranking.iloc[1]]}')

    qrels = vaswani_dataset.get_qrels()
    print('\nqrels ===> ')
    print(qrels.head())
    print(f'dtypes: qrels.iloc[0] => {[type(_) for _ in qrels.iloc[0]]}')
    print(f'dtypes: qrels.iloc[1] => {[type(_) for _ in qrels.iloc[1]]}')

    eval = pt.Utils.evaluate(ranking,qrels)
    print('\neval ===> ')
    print(eval)
    print('#> Pyterrier sanity check: end  : '+'#'*30)

    print('\n\n')

#> Pyterrier sanity check: start: ##############################                                                                                                                   
PyTerrier 0.6.0 has loaded Terrier 5.5 (built by craigmacdonald on 2021-05-20 13:12)                                                                                               
No etc/terrier.properties, using terrier.default.properties for bootstrap configuration.                                                                                           
Number of documents: 11429                                                                                                                                                         
Number of terms: 7756                                                                                                                                                              
Number of postings: 224573                                                                                                                                                         
Number of fields: 0                                                                                                                                                                
Number of tokens: 271581                                                                                                                                                           
Field names: []                                                                                                                                                                    
Positions:   false                                                                                                                                                                 


ranking ===>
  qid docno  rank      score
0   1  8172     0  13.746087
1   1  9881     1  12.352666
2   1  5502     2  12.178153
3   1  1502     3  10.993585
4   1  9859     4  10.271452
dtypes: ranking.iloc[0] => [<class 'str'>, <class 'str'>, <class 'numpy.int64'>, <class 'numpy.float64'>]
dtypes: ranking.iloc[1] => [<class 'str'>, <class 'str'>, <class 'numpy.int64'>, <class 'numpy.float64'>]

qrels ===>
  qid docno  label
0   1  1239      1
1   1  1502      1
2   1  4462      1
3   1  4569      1
4   1  5472      1
dtypes: qrels.iloc[0] => [<class 'str'>, <class 'str'>, <class 'numpy.int64'>]
dtypes: qrels.iloc[1] => [<class 'str'>, <class 'str'>, <class 'numpy.int64'>]

eval ===>
{'map': 0.29090543005529873, 'ndcg': 0.6153667539666847}
#> Pyterrier sanity check: end  : ##############################

    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrels', dest='qrels', default='data/2019qrels-pass.test.tsv')
    parser.add_argument('--qrels_exclude', type=str)
    parser.add_argument('--ranking', dest='ranking',)

    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    print(f'#> The # of samples in qrels = {len(qrels)}')
    print(qrels.head())
    
    if args.qrels_exclude:
        qrels_exclude = defaultdict(set)
        with open(args.qrels_exclude, mode='r', encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                qid, _, pid, rel = line.strip().split()
                # qrels_exclude[int(qid)].add(int(pid))
                qrels_exclude[qid.strip()].add(pid.strip())
    else:
        qrels_exclude = defaultdict(set)

    ranking = load_ranking(args.ranking, qrels_exclude)

    print('\n\n')

    if not pt.started():
        pt.init()
    
    # {
    #     'map','map_cut_10','map_cut_100','map_cut_1000','map_cut_15','map_cut_20','map_cut_200','map_cut_30','map_cut_5','map_cut_500',
    #     'ndcg','ndcg_cut_10','ndcg_cut_100','ndcg_cut_1000','ndcg_cut_15','ndcg_cut_20','ndcg_cut_200','ndcg_cut_30','ndcg_cut_5','ndcg_cut_500',
    #     'recall_10','recall_100','recall_1000','recall_15','recall_20','recall_200','recall_30','recall_5','recall_500',
    # }
    # eval = pt.Utils.evaluate(ranking, qrels, metrics=[
    #     "ndcg_cut_10","ndcg_cut_200",
    #     "map_cut_1000", 
    #     "recall_100","recall_200","recall_500","recall_1000",
    # ])
    # print(eval)
    

    from pyterrier.measures import RR, nDCG, AP, NumRet, R, P
    # RR: [Mean] Reciprocal Rank ([M]RR)
    # nDCG: The normalized Discounted Cumulative Gain (nDCG).
    # AP: The [Mean] Average Precision ([M]AP).
    # R: Recall@k (R@k).
    from pandas import DataFrame
    """ (from ColBERT-PRF)
     we report mean reciprocal rank (MRR) and normalised
    discounted cumulative gain (NDCG) calculated at rank 10, as well
    as Recall and Mean Average Precision (MAP) at rank 1000 [8]. For
    the MRR, MAP and Recall metrics, we treat passages with label
    grade 1 as non-relevant, following [7, 8].
    """
    eval = pt.Utils.evaluate(ranking, qrels, 
        metrics=[
            nDCG@10, nDCG@25, nDCG@50, nDCG@100, nDCG@200, nDCG@500, nDCG@1000, 
            R(rel=2)@3, R(rel=2)@5, R(rel=2)@10, R(rel=2)@25, R(rel=2)@50, R(rel=2)@100, R(rel=2)@200, R(rel=2)@1000,
            P(rel=2)@1, P(rel=2)@3, P(rel=2)@5, P(rel=2)@10, P(rel=2)@25, P(rel=2)@50, P(rel=2)@100, P(rel=2)@200, P(rel=2)@1000,
            AP(rel=2)@1000, RR(rel=2)@10, 
            NumRet, "num_q",
        ],
        # These measures are from "https://github.com/terrierteam/ir_measures/tree/f6b5dc62fd80f9e4ca5678e7fc82f6e8173a800d/ir_measures/measures"
    )
    print(eval)

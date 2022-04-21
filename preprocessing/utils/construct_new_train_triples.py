import argparse
from collections import defaultdict, OrderedDict
import ujson
import numpy as np
from tqdm import tqdm
import random
import time

from colbert.evaluation.loaders import load_qrels

def load_hn(path, k=100):
    print(f'#> Load {path}')
    qid_to_hn = {}
    with open(path) as ifile:
        for i_line, line in enumerate(ifile):
            
            qid, topk_tuples = line.strip().split('\t')

            qid = int(qid)
            topk_tuples = ujson.loads(topk_tuples)
            
            topk_tuples = topk_tuples[:k]
            qid_to_hn[qid] = [pid for pid, score in topk_tuples]

            if i_line < 10:
                print(f'\tpid for {qid} => {random.choice(qid_to_hn[qid])}')

    return qid_to_hn

if __name__=='__main__':
    random.seed(12345)

    parser = argparse.ArgumentParser("Construct new train triples using and hard negatives (HN).")

    parser.add_argument('--hn', dest='hn', required=True, help="Path to hard negatives.")
    parser.add_argument('--hn_topk', type=int, default=100, help="Top-k ranked PIDs for hard negatives.")
    parser.add_argument('--qrels', type=str, default='data/qrels.train.tsv', help="Path to labeled positives.")
    parser.add_argument('--output', dest='output', type=str, default='data/nn.triples.train.small.ids.jsonl', help="Output path")
    parser.add_argument('--n_triples', dest='n_triples', type=int, default=40000000)
    # wc -l triples.train.small.ids.jsonl 
    # 39780811 data/triples.train.small.ids.jsonl

    args = parser.parse_args()
    
    st = time.time()
    positives = load_qrels(qrels_path=args.qrels)
    for qid in positives:
        positives[qid] = set(positives[qid])
    queries_in_order = list(positives.keys())
    print(f'#> Elapsed time for loading positives: {time.time()-st}')
    
    st = time.time()
    negatives = load_hn(path=args.hn, k=args.hn_topk)
    assert len( set(negatives.keys()) - set(positives.keys()) ) == 0 and len( set(positives.keys()) - set(negatives.keys()) ) == 0
    for qid in negatives:
        negatives[qid] = [pid for pid in negatives[qid] if pid not in positives[qid]] 
    print(f'#> Elapsed time for loading negatives: {time.time()-st}')
    
    print()
    num_positives = [len(_) for _ in positives.values()]
    print(f'#> The # of positives: Min {np.min(num_positives)}, Max {np.max(num_positives)}, Mean {np.mean(num_positives):.2f}, Median {np.median(num_positives)}')
    num_negatives = [len(_) for _ in negatives.values()]
    print(f'#> The # of negatives: Min {np.min(num_negatives)}, Max {np.max(num_negatives)}, Mean {np.mean(num_negatives):.2f}, Median {np.median(num_negatives)}')
    del num_positives, num_negatives
    print()

    st = time.time()
    print(f'#> Construct new train triples: list of (query ID, positive psg ID, negative psg ID)')
    triples = [(qid, ppid, npid) for qid in queries_in_order for ppid in positives[qid] for npid in negatives[qid]]
    print(f'#> Elapsed time for constructing triples: {time.time()-st}')
    
    st = time.time()
    print(f'#> Shuffle ')
    random.shuffle(triples)
    print(f'#> Elapsed time for shuffling triples: {time.time()-st}')
    
    if args.n_triples > -1:
        print(f'#> Only retain top-{args.n_triples} triples to save disk (training often be early terminated before using all train triples).')
        triples = triples[:args.n_triples]

    st = time.time()
    print(f'#> Save triples to {args.output}')
    with open(args.output, 'w') as outfile:
        for qid, ppid, npid in triples:
            outfile.write(ujson.dumps([qid, ppid, npid])+'\n')
    print(f'#> Elapsed time for saving triples: {time.time()-st}')
    print(f'#> output: \t\t {args.output}')

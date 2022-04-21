import csv
import argparse
import collections
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrels', default='/workspace/DataCenter/PassageRanking/MSMARCO/qrels.train.tsv')
    
    args = parser.parse_args()

    qrels = collections.defaultdict(set)
    with open(args.qrels) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            qid, _, pid, _ = map(int, row)
            qrels[qid].add(pid)
    
    n_golds = [len(pids) for pids in qrels.values()]
    n_golds = np.array(n_golds, dtype=np.int64)

    print(f'# of positive passages per query = min {n_golds.min()}, max {n_golds.max()}, mean {n_golds.mean():2.3f}, medial {np.median(n_golds)}')



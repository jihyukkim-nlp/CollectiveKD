from argparse import ArgumentParser
import json
from typing_extensions import OrderedDict
import numpy as np

from colbert.evaluation.loaders import load_qrels
from colbert.utils.utils import print_message

if __name__=='__main__':
    
    parser = ArgumentParser("Pseudo-label unlabeled positives (UP) from top-ranked passages.")
    
    parser.add_argument('--ranking_jsonl', type=str, required=True)
    parser.add_argument('--thr', type=float, default=-1.0)
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--labeled_qrels', type=str)
    parser.add_argument('--output', required=True)
    parser.add_argument('--add_labeled_positive', action='store_true')

    args = parser.parse_args()
    
    qrels = load_qrels(args.labeled_qrels)
    # qrels = {k:set(v) for k, v in qrels.items()}
    _qrels = OrderedDict()
    for qid, pids in qrels.items():
        _qrels[qid] = set(pids)
    qrels = _qrels

    print_message(f'#> ranking_jsonl: \t{args.ranking_jsonl}')
    print_message(f'#> output       : \t{args.output}')
    print_message(f'#> topk {args.topk}, thr {args.thr}')

    nprd = []

    with open(args.ranking_jsonl) as infile, open(args.output, 'w') as outfile:

        for line_idx, line in enumerate(infile):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)
        
            qid, topk_tuples = line.strip().split('\t')

            qid = int(qid)
            topk_tuples = json.loads(topk_tuples) # List[Tuple(int, float)] = list of (pid, score)
            # print(f'(org) topk_tuples: {topk_tuples[:10]}')
            
            # remove labeled positives
            topk_tuples = [tup for tup in topk_tuples if tup[0] not in qrels[qid]]

            # select pseudo-positives using top-k cut-off
            if args.topk > -1:
                topk_tuples = topk_tuples[:args.topk]
            
            # select pseudo-positives using score threshold
            ups = [pid for pid, score in topk_tuples if score >= args.thr]
            
            # outfile.write(f'{qid}\t{json.dumps(ups)}\n')
            
            """
            qid     0       pid     1

            1185869 0       0       1
            1185868 0       16      1
            597651  0       49      1
            403613  0       60      1
            1183785 0       389     1
            """
            for pid in ups:
                outfile.write(f'{qid}\t0\t{pid}\t1\n')

            nprd.append(len(ups))
        
        print_message(f'#> The # of positives: Min {np.min(nprd)}, Max {np.max(nprd)}, Mean {np.mean(nprd):.2f}, Median {np.median(nprd)}')
        
        # Add labeled positives
        if args.add_labeled_positive:
            print_message(f'#> Add labeled positives: {args.labeled_qrels}')
            print_message(f'#> \"{args.output}\" contains both pseudo-positives and labeled positives.')
            for qid, pids in qrels.items():
                for pid in pids:
                    outfile.write(f'{qid}\t0\t{pid}\t1\n')
        else:
            print_message(f'#> \"{args.output}\" contains only pseudo-positives.')




        
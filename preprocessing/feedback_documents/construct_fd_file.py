import argparse
import os
import json

from collections import OrderedDict

from colbert.evaluation.loaders import load_qrels


def load_prf_pids(topK_path, fb_docs=3):
    # topK_path: ranking.tsv

    topK_pids = OrderedDict()

    print("#> Loading the top-k PIDs per query from", topK_path, "...")

    with open(topK_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)

            qid, pid, rank, score = line.strip().split('\t')
            
            if int(rank) <= fb_docs:
                qid, pid = int(qid), int(pid)
                topK_pids[qid] = topK_pids.get(qid, [])
                topK_pids[qid].append(pid)

        print()

    assert all(len(topK_pids[qid]) == len(set(topK_pids[qid])) for qid in topK_pids)

    Ks = [len(topK_pids[qid]) for qid in topK_pids]
    print("#> max(Ks) =", max(Ks), ", avg(Ks) =", round(sum(Ks) / len(Ks), 2))
    print("#> Loaded the top-k per query for", len(topK_pids), "unique queries.\n")

    return topK_pids

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--ranking', type=str) # used as pseudo-relevance feedback
    parser.add_argument('--qrels', type=str,) # used as relevance feedback
    parser.add_argument('--fb_docs', type=int, default=3, help="The number of feedback documents used for query expansion.")
    parser.add_argument('--output', type=str, default="fd.jsonl")

    args = parser.parse_args()

    # make directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    assert args.ranking or args.qrels, 'Either ``ranking`` (PRF) or ``qrels`` (RF) must be given.'

    # load fd (feedback documents)
    
    # PRF (pseudo-relevance feedback)
    if args.ranking:
        prf_fd_dict = load_prf_pids(topK_path=args.ranking, fb_docs=args.fb_docs)
    else:
        prf_fd_dict = None
    
    # RF (relevance feedback, i.e., labeled relevant documents)
    if args.qrels:
        rf_fd_dict = load_qrels(qrels_path=args.qrels)
        for qid in list(rf_fd_dict.keys()):
            rf_fd_dict[qid] = rf_fd_dict[qid][:args.fb_docs]
    else:
        rf_fd_dict = None
    
    # Merge the two, if possible
    if (prf_fd_dict is None) and (rf_fd_dict is not None):
        fd_dict = rf_fd_dict
    elif (prf_fd_dict is not None) and (rf_fd_dict is None):
        fd_dict = prf_fd_dict
    elif (prf_fd_dict is not None) and (rf_fd_dict is not None):
        
        fd_dict = rf_fd_dict
        
        for qid in list(fd_dict.keys()):
            n_add_fd = args.fb_docs - len(fd_dict[qid])
            if n_add_fd > 0:
                rf_pids = set(fd_dict[qid])
                for pid in prf_fd_dict[qid]:
                    if pid not in rf_pids:
                        fd_dict[qid].append(pid)
                        if len(fd_dict[qid]) == args.fb_docs:
                            break
    # fd_dict: OrderedDict[ qid -> list of pids]

    for qid, pids in fd_dict.items():
        assert len(pids) <= args.fb_docs
    
    with open(args.output, 'w') as outfile:
        for qid, pids in fd_dict.items():
            outfile.write(f'{qid}\t{json.dumps(pids)}\n')
    
    print(f'\n\n\n Outfile: \t\t{args.output}')



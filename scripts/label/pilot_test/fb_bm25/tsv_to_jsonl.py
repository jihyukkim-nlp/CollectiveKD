import argparse, json
from collections import OrderedDict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv')
    parser.add_argument('--output')
    
    args = parser.parse_args()

    ranking = OrderedDict()

    with open(args.tsv, 'r') as ifile, open(args.output, 'w') as ofile:
        for line_idx, line in enumerate(ifile):
            qid, pid, rank = line.strip().split()
            
            ranking[qid] = ranking.get(qid, [])
            ranking[qid].append(pid)
            
        for qid in ranking:
            pids = [int(_) for _ in ranking[qid]]
            scores = [float(_) for _ in list(range(len(pids)))[::-1]]
            pairs = list(zip(pids, scores))
            ofile.write(f'{int(qid)}\t{json.dumps(pairs)}\n')
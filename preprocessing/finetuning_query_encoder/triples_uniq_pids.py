import argparse
import json

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--triples')

    args = parser.parse_args()

    pid_set = set()
    with open(args.triples) as ifile:
        for line_idx, line in enumerate(ifile):
            qid, *pids = json.loads(line)
            pid_set.update(pids)
    print(f'# of uniq pids: {len(pid_set)}')
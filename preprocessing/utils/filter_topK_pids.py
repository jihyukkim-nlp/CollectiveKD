from collections import defaultdict
from colbert.utils.utils import print_message
import os

def load_topK_pids_filtered(topK_path, filtered_qid_set):
    topK_pids = defaultdict(list)

    print_message("#> Loading the top-k PIDs per query from", topK_path, "...", f' : retained # of queries after filtering = {len(filtered_qid_set)}')

    with open(topK_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)

            qid, pid, *rest = line.strip().split('\t')
            qid, pid = int(qid), int(pid)

            if qid in filtered_qid_set:
                topK_pids[qid].append(pid)

        print()

    assert all(len(topK_pids[qid]) == len(set(topK_pids[qid])) for qid in topK_pids)
    
    Ks = [len(topK_pids[qid]) for qid in topK_pids]

    print_message("#> max(Ks) =", max(Ks), ", avg(Ks) =", round(sum(Ks) / len(Ks), 2))
    print_message("#> Loaded the top-k per query for", len(topK_pids), "unique queries.\n")

    return topK_pids


def filter_by_qrels(qid_dict, qrels):
    qids_in_order = list(qid_dict.keys())
    for qid in qids_in_order:
        if qid not in qrels:
            del qid_dict[qid]
    print_message(f'#> (filter_by_qrels) Reduce the # of queries: {len(qids_in_order)} -> {len(qid_dict)}')
def filter_by_queries(qid_dict, queries):
    qids_in_order = list(qid_dict.keys())
    for qid in qids_in_order:
        if qid not in queries:
            del qid_dict[qid]
    print_message(f'#> (filter_by_queries) Reduce the # of queries: {len(qids_in_order)} -> {len(qid_dict)}')


import argparse
if __name__=='__main__':
    """ Filter top-k PIDs.
    1. Get candidate top-k PIDs from ``--topk``.
    2. Get target queries from ``--queries``.
    3. Filter top-k PIDs for queries that are included in target queries.
    
    Usage: To filter PIDs from ANN search, using sub-queries.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', dest='topK', required=True, help="Top-K PIDs from ANN search")
    # parser.add_argument('--queries', dest='queries', required=True, help="queries for filtering")
    # parser.add_argument('--filtered_topk', dest='filtered_topk', required=True, help="Output path")
    parser.add_argument('--queries', dest='queries', required=True, help="queries for filtering", nargs="+")
    parser.add_argument('--filtered_topk', dest='filtered_topk', required=True, help="Output path", nargs="+")

    args = parser.parse_args()

    topK_pids = defaultdict(list)
    
    def load_qid_set(path):
        print(f'\n\n\nLoad queries from "{path}"')
        qid_set = set()
        with open(path, 'r', encoding='utf-8') as ifile:
            for line in ifile:
                qid = line.strip().split('\t')[0]
                qid_set.add(int(qid))
        print(f'#> The # of queries={len(qid_set)}\n')
        return qid_set

    print(f'filter topK_ids: \n\tfrom {args.topK} \n\t==> to {os.path.dirname(args.filtered_topk[0])}')
    query_file_index = -1
    query_file_qidset = set()
    outfile = None
    with open(args.topK) as f:
        for line_idx, line in enumerate(f):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)

            qid, pid, *rest = line.strip().split('\t')
            qid, pid = int(qid), int(pid)

            if qid not in query_file_qidset:
                print()
                # Close the current outfile
                if (outfile is not None):
                    outfile.close()
                # Increase query_file_index by one
                query_file_index += 1
                # Update qid_set
                query_file_qidset = load_qid_set(path=args.queries[query_file_index])
                # Open new outfile
                outfile = open(args.filtered_topk[query_file_index], 'w')
                print(f'Write filtered pids to "{args.filtered_topk[query_file_index]}"')
                print()

            outfile.write(line)
    
    outfile.close()


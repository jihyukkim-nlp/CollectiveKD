from collections import defaultdict, OrderedDict
from typing import DefaultDict
from colbert.utils.utils import print_message
import os
import random
import csv
import ujson
import numpy as np

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.indexing.faiss import get_faiss_index_name

from colbert.evaluation.loaders import load_colbert, load_collection, load_qrels, load_queries, load_topK_pids

from colbert.labeling.batch_reranking import batch_rerank

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

def load_fb(path, k=3):
    print(f'#> Load {path}')
    fb_pids = {}
    with open(path) as ifile:
        for i_line, line in enumerate(ifile):
            
            qid, topk_tuples = line.strip().split('\t')

            qid = int(qid)
            topk_tuples = ujson.loads(topk_tuples)
            
            topk_tuples = topk_tuples[:k]
            if isinstance(topk_tuples[0], int):
                fb_pids[qid] = topk_tuples # consisting of only pids, without scores
            else:
                assert isinstance(topk_tuples[0], tuple) or isinstance(topk_tuples[0], list)
                fb_pids[qid] = [pid for pid, score in topk_tuples]
            
    return fb_pids

def main():
    random.seed(12345)

    parser = Arguments(description='Retrieving unlabeled positive passages using ColBERT-RF.')

    parser.add_model_parameters() # similarity, dim, query/doc_maxlen, mask-punctutations
    parser.add_model_inference_parameters() # checkpoint, bsize, amp
    # parser.add_reranking_input() # topK, (add_ranking_input; queries, collection, qrels)
    parser.add_argument('--queries', dest='queries', default=None)
    parser.add_argument('--collection', dest='collection', default=None)
    parser.add_argument('--qrels', dest='qrels', default=None)
    parser.add_argument('--topk', dest='topK')
    parser.add_index_use_input() # index_root, index_name

    # Arguments for ANN search
    parser.add_argument('--faiss_name', dest='faiss_name', default=None, type=str)
    parser.add_argument('--faiss_depth', dest='faiss_depth', default=1024, type=int)
    parser.add_argument('--nprobe', dest='nprobe', default=10, type=int)

    # Arguments for Exact-NN search (re-ranking)
    parser.add_argument('--step', dest='step', default=1, type=int)
    parser.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser.add_argument('--log-scores', dest='log_scores', default=False, action='store_true') #
    parser.add_argument('--batch', dest='batch', default=False, action='store_true') # 
    parser.add_argument('--depth', dest='depth', default=1000, type=int)

    # Arguments for query expansion using ColBERT-PRF
    parser.add_argument('--prf', dest='prf', default=False, action='store_true', help="Leveraging pseudo-relevance feedback") 
    parser.add_argument('--expansion_pt', type=str, help="/path/to/label.py/expansion.pt") # experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/expansion.pt
    parser.add_argument('--fb_ranking', dest='fb_ranking', help="/path/to/label.py/*/ranking.jsonl")
    parser.add_argument('--prepend_rf', action='store_true', help="Whether to prepend labeled documents in front of top-ranked documents, as feedback documents.")

    parser.add_argument('--fb_docs', dest='fb_docs', default=3, type=int, help="Only valid for pseudo-relevance feedback")
    parser.add_argument('--fb_clusters', dest='fb_clusters', default=24, type=int)
    parser.add_argument('--fb_k', dest='fb_k', default=10, type=int)
    parser.add_argument('--beta', dest='beta', default=0.5, type=float)
    parser.add_argument('--kmeans_init', default='avg_step_position', type=str, choices=['top1_step_position', 'avg_step_position', 'random'])

    parser.add_argument('--score_by_range', action='store_true')

    parser.add_argument('--expansion_only', action='store_true')
    
    args = parser.parse()

    if args.prf:
        assert args.fb_ranking and os.path.exists(args.fb_ranking)
        print_message(f'prf =====> fb_docs: {args.fb_docs}, fb_clusters: {args.fb_clusters}, fb_k: {args.fb_k}, beta: {args.beta}')
        args.fb_ranking = load_fb(args.fb_ranking, args.fb_docs)
        _n_fb_pids = list([len(pids) for qid, pids in args.fb_ranking.items()])
        print_message(f'#> The number of feedback documents for {len(_n_fb_pids)} queries: min {np.min(_n_fb_pids)}, max {np.max(_n_fb_pids)}, mean {np.mean(_n_fb_pids):.3f}')
            
    
    if not args.expansion_only:
        assert args.topK and os.path.exists(args.topK)

    # tag = os.path.basename(args.queries)
    # if tag.endswith('.tsv'):
    #     tag = tag[:-len('.tsv')] # queries.train.reduced

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    with Run.context():

        # Load data: queries, qrels, (unordered) topK_pids (being reranked)
        args.queries = load_queries(args.queries)
        args.qrels = load_qrels(args.qrels)
        # filter_by_qrels(args.queries, qrels=args.qrels)
        # filter_by_queries(args.qrels, queries=args.queries)

        qids_in_order = list(args.queries.keys())
        # qids_in_order: List[int] = list of qids (in order)

        if not args.expansion_only:
            args.topK_pids, args.qrels = load_topK_pids(args.topK, qrels=args.qrels)
            # filter_by_qrels(args.topK_pids, qrels=args.qrels)
            # filter_by_queries(args.topK_pids, queries=args.queries)
            # assert len(args.queries) == len(args.qrels) == len(args.topK_pids), f'len(args.queries)={len(args.queries)}, len(args.qrels)={len(args.qrels)}, len(args.topK_pids)={len(args.topK_pids)}'
        else:
            # args.topK_pids = {qid:[] for qid in args.qrels}
            args.topK_pids = {qid:[] for qid in args.queries}
            args.topK_pids.update({qid:[] for qid in args.qrels})

        # args.queries: OrderedDict[int:str] = qid -> query
        # args.qrels: OrderedDict[int:List[int]] = qid -> pids of positive passages
        # args.topK_pids: Dict[int:List[int]] = qid -> topK pids
        
        args.collection = load_collection(args.collection)
        
        args.index_path = os.path.join(args.index_root, args.index_name)
        
        if args.faiss_name is not None:
            args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
        else:
            args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))

        args.colbert, args.checkpoint = load_colbert(args)

        # Rank unordered pids
        if args.batch:
            batch_rerank(args)
        else:
            #TODO
            raise NotImplementedError
            rerank(args)

        # Construct ranked_topK_pids
        ranked_topK_pids = OrderedDict()
        
        def update_ranked_topK_pids(file):
            print_message(f'#> Load {file}')
            with open(file) as ifile:
                reader  = csv.reader(ifile, delimiter='\t')
                for row in reader:
                    qid, pid, rank, score = row
                    qid, pid = int(qid), int(pid)
                    rank, score = int(rank), float(score)
                    ranked_topK_pids[qid] = ranked_topK_pids.get(qid, [])
                    ranked_topK_pids[qid].append((pid, score))
        
        def sanity_check():
            print_message(f'#> Sanity check ===> len(ranked_topK_pids);{len(ranked_topK_pids)} == len(qids_in_order);{len(qids_in_order)}')
            assert len(ranked_topK_pids) == len(qids_in_order)
            for qid1, qid2 in zip(ranked_topK_pids, qids_in_order):
                assert qid1 == qid2
        
        def save_ranked_topK_pids():
            outfile_path = os.path.join(Run.path, f'ranking.jsonl')
            print_message(f'#> Save save_ranked_topK_pids into \n\n\t{outfile_path}\n')
            with open(outfile_path, 'w') as outfile:
                for qid in ranked_topK_pids:
                    outfile.write(f'{qid}\t{ujson.dumps(ranked_topK_pids[qid][:args.depth])}\n')
        
        print_message(f'#> Update ranked_topK_pids')
        ranking_logger_file = os.path.join(Run.path, 'ranking.tsv')
        update_ranked_topK_pids(ranking_logger_file)
        
        sanity_check()
        save_ranked_topK_pids()
        
        # os.remove(ranking_logger_file)

        print(f'\n#> Process is done !')




if __name__=='__main__':
    # sanity_check()
    main()
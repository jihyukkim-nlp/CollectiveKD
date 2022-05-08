import os
import time
import torch
import queue
import threading
import numpy as np
from tqdm import tqdm

from collections import defaultdict

from colbert.utils.runs import Run
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, flatten, zipstar
from colbert.indexing.loaders import get_parts

from colbert.modeling.colbert_prf import ColbertPRF
# from colbert.modeling.colbert_prf_qdmaxsim import ColbertPrfQDMaxsim

from colbert.training.lazy_batcher import load_expansion_pt

def score_by_range(positions, loaded_parts, all_query_embeddings, all_query_rankings, all_pids, all_query_weights):
    # all_query_embeddings: float tensor, size (n_queries, dim, query_maxlen)
    # all_query_weights: float tensor, size (n_queries, query_maxlen)

    print_message("#> Sorting by PID..")
    all_query_indexes, all_pids = zipstar(all_pids)
    sorting_pids = torch.tensor(all_pids).sort()
    all_query_indexes, all_pids = torch.tensor(all_query_indexes)[sorting_pids.indices], sorting_pids.values

    range_start, range_end = 0, 0

    for offset, endpos in positions:
        print_message(f"#> Fetching parts {offset}--{endpos} from queue..")
        index = loaded_parts.get()

        print_message(f"#> Filtering PIDs to the range {index.pids_range}..")
        range_start = range_start + (all_pids[range_start:] < index.pids_range.start).sum()
        range_end = range_end + (all_pids[range_end:] < index.pids_range.stop).sum()

        pids = all_pids[range_start:range_end]
        query_indexes = all_query_indexes[range_start:range_end]

        print_message(f"#> Got {len(pids)} query--passage pairs in this range.")

        if len(pids) == 0:
            continue

        print_message(f"#> Ranking in batches the pairs #{range_start} through #{range_end}...")
        scores = index.batch_rank(all_query_embeddings=all_query_embeddings, query_indexes=query_indexes, pids=pids, sorted_pids=True, all_query_weights=all_query_weights)

        for query_index, pid, score in zip(query_indexes.tolist(), pids.tolist(), scores):
            all_query_rankings[0][query_index].append(pid)
            all_query_rankings[1][query_index].append(score)

def score_all(index, all_query_embeddings, all_query_rankings, all_pids, all_query_weights):
    # all_query_embeddings: float tensor, size (n_queries, dim, query_maxlen)
    # all_query_weights: float tensor, size (n_queries, query_maxlen)

    print_message("#> Sorting by PID..")
    all_query_indexes, all_pids = zipstar(all_pids)
    sorting_pids = torch.tensor(all_pids).sort()
    all_query_indexes, all_pids = torch.tensor(all_query_indexes)[sorting_pids.indices], sorting_pids.values

    pids = all_pids
    query_indexes = all_query_indexes

    scores = index.batch_rank(all_query_embeddings=all_query_embeddings, query_indexes=query_indexes, pids=pids, sorted_pids=True, all_query_weights=all_query_weights)

    for query_index, pid, score in zip(query_indexes.tolist(), pids.tolist(), scores):
        all_query_rankings[0][query_index].append(pid)
        all_query_rankings[1][query_index].append(score)

def save_expansion(queries, all_exp_embeddings, all_exp_weights, all_exp_tokens):
    _exp_save_path=os.path.join(Run.path, 'expansion.pt')
    qid_to_embs = {qid:val for qid, val in zip(queries, all_exp_embeddings)}
    qid_to_weights = {qid:val for qid, val in zip(queries, all_exp_weights)}
    qid_to_tokens = {qid:val for qid, val in zip(queries, all_exp_tokens)}
    torch.save({'qid_to_embs':qid_to_embs, 'qid_to_weights': qid_to_weights, 'qid_to_tokens': qid_to_tokens}, _exp_save_path)
    print_message(f'#> Save expansion embeddings/weights/tokens into: \t{_exp_save_path}')
    print_message(f'#> \t expansion embeddings (size {tuple(all_exp_embeddings.size())}, dtype {all_exp_embeddings.dtype}, device {all_exp_embeddings.device})')
    print_message(f'#> \t expansion weights    (size {tuple(all_exp_weights.size())}, dtype {all_exp_weights.dtype}, device {all_exp_weights.device})')
    print_message(f'#> \t expansion tokens     (size {len(all_exp_tokens)})')

def batch_qe(args, colbert_prf, queries, fb_pids, all_query_embeddings, all_query_weights=None):
    
    with torch.no_grad():
        if args.fb_k > 0 and args.beta > 0.0:
            print_message(f'#> query expansion from feedback documents; (dim {args.dim}, org_qlen {all_query_embeddings.size(2)}) -> (dim {args.dim}, org_qlen {all_query_embeddings.size(2)} + fb_k {args.fb_k}) (beta={args.beta})')
            _n_fb_pids = list([len(pids) for qid, pids in fb_pids.items()])
            print_message(f'#> The number of feedback documents for {len(_n_fb_pids)} queries: min {np.min(_n_fb_pids)}, max {np.max(_n_fb_pids)}, mean {np.mean(_n_fb_pids):.3f}')
            
            # rel_docs_pids = [fb_pids[qid] for qid in queries]
            rel_docs_pids = [fb_pids.get(qid, []) for qid in queries] # there can be empty feedback documents
            # rel_docs_pids: List[List[int]] = for each qid (outer list), list of pids of relevant documents (inner list)

            # Expand query
            print_message(f'#> Expand query')
            _offset = 0
            all_exp_embeddings = torch.zeros(len(all_query_embeddings), args.fb_k, args.dim, dtype=all_query_embeddings.dtype, device=all_query_embeddings.device)
            all_exp_weights = torch.zeros(len(all_query_embeddings), args.fb_k, dtype=all_query_embeddings.dtype, device=all_query_embeddings.device)
            all_exp_tokens = []

            # Extract expansion query term from relevance feedback documents, for each query
            for query_index, _pids in enumerate(tqdm(rel_docs_pids)):
                if len(_pids) > 0:
                    _endpos = _offset + len(_pids)
                    
                    exp_embs, exp_weights, exp_tokens = colbert_prf.expand(
                        q_embs=all_query_embeddings[query_index].transpose(0,1), 
                        fb_pids=_pids,
                    )
                    # exp_embs: cpu, float-32 tensor (fb_k, dim) = expansion embeddings
                    # exp_weights: cpu, float-32 tensor (fb_k) = weights for the expansion embeddings
                    # exp_tokens: List[str] = list of expansion tokens (len=fb_k)

                else:
                    _endpos = _offset + 1
                    exp_embs = torch.zeros(args.fb_k, args.dim, dtype=all_query_embeddings.dtype, device=all_query_embeddings.device)
                    exp_weights = torch.zeros(args.fb_k, dtype=all_query_embeddings.dtype, device=all_query_embeddings.device)
                    exp_tokens = ["[EmptyFeedback]"]*args.fb_k

                all_exp_embeddings[query_index, :len(exp_embs)] = exp_embs
                all_exp_weights[query_index, :len(exp_weights)] = exp_weights
                all_exp_tokens.append(exp_tokens)

                _offset = _endpos
            
            # Save results
            save_expansion(queries, all_exp_embeddings, all_exp_weights, all_exp_tokens)
            
            # Expand query embeddings, along with corresponding weights
            all_exp_embeddings = all_exp_embeddings.permute(0, 2, 1).contiguous()
            # all_exp_embeddings: float32 tensor, size (n_queries, dim, fb_k), on cpu device 
            all_query_embeddings = torch.cat((all_query_embeddings, all_exp_embeddings), dim=-1)
            # all_query_embeddings: float32 tensor, size (n_queries, dim, query_maxlen + fb_k), on cpu device 
            if (all_query_weights is None):
                all_query_weights = torch.cat((torch.ones(
                    all_query_embeddings.size(0), args.query_maxlen, dtype=all_exp_weights.dtype, device=all_exp_weights.device
                ), all_exp_weights), dim=1)
            else:
                all_query_weights = torch.cat((all_query_weights, all_exp_weights), dim=1)
            # all_query_weights: float16 tensor, size (n_queries, query_maxlen + fb_k), on cpu device 

        else:
            print_message(f'#> w/o query expansion from feedback documents: fb_k {args.fb_k}, beta {args.beta}')
            # without query expansion
            if (all_query_weights is None):
                all_query_weights = torch.ones(all_query_embeddings.size(0), args.query_maxlen, dtype=all_query_embeddings.dtype, device=all_query_embeddings.device)

    print('\n\n\n')
    print_message('#> Done!')
    print_message(f'all_query_embeddings (shape {all_query_embeddings.shape}, dtype {all_query_embeddings.dtype}, device {all_query_embeddings.device})')
    print_message(f'all_query_weights (shape {all_query_weights.shape}, dtype {all_query_weights.dtype}, device {all_query_weights.device})')
    print('\n\n\n')
    return all_query_embeddings, all_query_weights

def batch_rerank(args):

    inference = ModelInference(args.colbert, amp=args.amp)
    colbert_prf = ColbertPRF(args=args, inference=inference)

    queries, topK_pids = args.queries, args.topK_pids
    for qid in queries:
        """
        Since topK_pids is a defaultdict, make sure each qid *has* actual PID information (even if empty).
        """
        assert qid in topK_pids, qid

    # Encode the original query
    with torch.no_grad():
        queries_in_order = list(queries.values())

        print_message(f"#> Encoding all {len(queries_in_order)} queries in batches...")

        all_query_embeddings = inference.queryFromText(queries_in_order, bsize=512, to_cpu=True) # n_queries, query_maxlen, dim
        all_query_embeddings = all_query_embeddings.to(dtype=torch.float32).permute(0, 2, 1).contiguous() # n_queries, dim, query_maxlen

    if args.prf:
        if args.expansion_pt and os.path.exists(args.expansion_pt):
            qexp_embs, qexp_wts = load_expansion_pt(args.expansion_pt)
            all_exp_embeddings = torch.stack([qexp_embs[qid] for qid in queries], dim=0)
            all_exp_weights = torch.stack([qexp_wts[qid] for qid in queries], dim=0)
            print_message(f'#> Load expansion embeddings/weights/tokens from: \t{args.expansion_pt}')
            print_message(f'#> \t all_exp_embeddings (shape {all_exp_embeddings.shape}, dtype {all_exp_embeddings.dtype}, device {all_exp_embeddings.device}) ')
            print_message(f'#> \t all_exp_weights (shape {all_exp_weights.shape}, dtype {all_exp_weights.dtype}, device {all_exp_weights.device}) ')
            
            # Expand query embeddings, along with corresponding weights
            all_exp_embeddings = all_exp_embeddings.permute(0, 2, 1).contiguous()
            # all_exp_embeddings: float32 tensor, size (n_queries, dim, fb_k), on cpu device 
            all_query_embeddings = torch.cat((all_query_embeddings, all_exp_embeddings), dim=-1)
            # all_query_embeddings: float32 tensor, size (n_queries, dim, query_maxlen + fb_k), on cpu device 
            all_query_weights = torch.cat((torch.ones(
                all_query_embeddings.size(0), args.query_maxlen, dtype=all_exp_weights.dtype, device=all_exp_weights.device
            ), all_exp_weights), dim=1)
            # all_query_weights: float16 tensor, size (n_queries, query_maxlen + fb_k), on cpu device 
            print_message(f'#> \t all_query_embeddings (shape {all_query_embeddings.shape}, dtype {all_query_embeddings.dtype}, device {all_query_embeddings.device})')
            print_message(f'#> \t all_query_weights (shape {all_query_weights.shape}, dtype {all_query_weights.dtype}, device {all_query_weights.device})')
        else:
            print_message(f'#> Do not load expansion embeddings/weights/tokens from explicit relevance feedback, i.e., leveraging only pseudo-relevance feedback')
            all_query_weights = None    
        
        if args.prepend_rf:
            print_message(f'#> Prepend labeled relevant documents in front of the top-ranked documents, as feedback documents')
            # Prepend labeled documents in front of top-ranked documents, for feedback documents
            fb_ranking_with_rf = {}
            
            for qid, pids in args.fb_ranking.items():
                
                if qid in args.qrels:
                    labeled_pids = args.qrels[qid]
                    fb_ranking_with_rf[qid] = list(labeled_pids)[:args.fb_docs]
                    if len(fb_ranking_with_rf[qid]) < args.fb_docs:
                        _bucket = set(fb_ranking_with_rf[qid])
                        _prf_pids = [pid for pid in pids if pid not in _bucket]
                        _prf_pids = _prf_pids[:args.fb_docs-len(fb_ranking_with_rf[qid])]
                        fb_ranking_with_rf[qid].extend(_prf_pids)
                else:
                    fb_ranking_with_rf[qid] = pids
            
            # sanity check
            assert len(fb_ranking_with_rf) == len(args.fb_ranking)
            for qid, pids in fb_ranking_with_rf.items():
                assert len(pids)==args.fb_docs
            
            args.fb_ranking = fb_ranking_with_rf
        
        # Expand query embeddings, along with weights, using pseudo-relevance feedback
        all_query_embeddings, all_query_weights = batch_qe(args, colbert_prf, queries, args.fb_ranking, all_query_embeddings, all_query_weights)
    else:
        # Expand query embeddings, along with weights, using explicit relevance feedback
        all_query_embeddings, all_query_weights = batch_qe(args, colbert_prf, queries, args.qrels, all_query_embeddings)

    if args.expansion_only:
        print_message(f'#> ``--expansion_only`` is given. ====> do not rank documents; exit()')
        exit()

    # Re-rank using expanded query token embeddings along with weights for expansion embeddings
    all_pids = flatten([[(query_index, pid) for pid in topK_pids[qid]] for query_index, qid in enumerate(queries)])
    all_query_rankings = [defaultdict(list), defaultdict(list)]
    print_message(f"#> Will process {len(all_pids)} query--document pairs in total.")
    with torch.no_grad():
        if not args.score_by_range:
            score_all(colbert_prf.index, all_query_embeddings, all_query_rankings, all_pids, all_query_weights)
        else:
            #TODO: score_by_range
            score_by_range(colbert_prf.positions, colbert_prf.loaded_parts, all_query_embeddings, all_query_rankings, all_pids, all_query_weights)




    ranking_logger = RankingLogger(Run.path, qrels=None, log_scores=args.log_scores)
    with ranking_logger.context('ranking.tsv', also_save_annotations=False) as rlogger:
        with torch.no_grad():
            for query_index, qid in enumerate(queries):
                if query_index % 1000 == 0:
                    print_message("#> Logging query #{} (qid {}) now...".format(query_index, qid))

                pids = all_query_rankings[0][query_index]
                scores = all_query_rankings[1][query_index]

                K = min(args.depth, len(scores))

                if K == 0:
                    continue

                scores_topk = torch.tensor(scores).topk(K, largest=True, sorted=True)

                pids, scores = torch.tensor(pids)[scores_topk.indices].tolist(), scores_topk.values.tolist()

                ranking = [(score, pid, None) for pid, score in zip(pids, scores)]
                assert len(ranking) <= args.depth, (len(ranking), args.depth)

                rlogger.log(qid, ranking, is_ranked=True, print_positions=[1, 2] if query_index % 100 == 0 else [])

    print(ranking_logger.filename)
    print_message('#> Done.\n')


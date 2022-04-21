import torch
import numpy as np

def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize):
    # queries: List[ Tuple(str, tensor (or None), tensor (or None) ] = list of (query, expansion embeddings, weights for the exp embs)
    # positives: List[ str ] = list of positive passages, for each query.
    # negatives: List[ List[str] ] = list of (N sampled negative passages), for each query.

    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    positives, positive_scores = zip(*positives)
    positives = list(positives)
    positive_scores = list(positive_scores)
    positive_scores_batches = _split_into_batches(positive_scores, bsize=bsize)
   
    _negatives = [[neg[0] for neg in negs] for negs in negatives]
    negative_scores = [[neg[1] for neg in negs] for negs in negatives]
    negatives = _negatives
    negative_scores_batches = _split_into_batches(negative_scores, bsize=bsize)

    N = len(queries)

    #!@ custom: kd_query_expansion
    queries, qexp_embs, qexp_wts = zip(*queries) 

    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    query_batches = _split_into_batches(Q_ids, Q_mask, bsize=bsize)
    
    if (qexp_embs[0] is not None): # i.e., kd_query_expansion==True
        qexp_embs = torch.stack(qexp_embs, dim=0)
        qexp_wts = torch.stack(qexp_wts, dim=0)
        qexp_batches = _split_into_batches(qexp_embs, qexp_wts, bsize=bsize)
    else:
        qexp_batches = [(None, None)] * len(query_batches)
    
    # #!@ custom: previous for one negative sample
    # D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives) 
    # D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)
    # (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask
    # positive_batches = _split_into_batches(positive_ids, positive_mask, bsize=bsize)
    # negative_batches = _split_into_batches(negative_ids, negative_mask, bsize=bsize)
    
    #!@ custom: for many negative samples
    D_ids, D_mask = doc_tokenizer.tensorize(positives + [passage for negative_passages in negatives for passage in negative_passages])
    
    positive_ids = D_ids[:N]
    positive_mask = D_mask[:N]
    # positive_ids, positive_mask: (``N, doc_maxlen``)
    positive_batches = _split_into_batches(positive_ids, positive_mask, bsize=bsize)

    negative_ids = D_ids[N:]
    negative_mask = D_mask[N:]
    n_negatives = len(negatives[0])
    negative_ids = negative_ids.view(N, n_negatives, -1)
    negative_mask = negative_mask.view(N, n_negatives, -1)
    # negative_ids, negative_mask: (``N, n_negatives, doc_maxlen``)
    negative_batches = _split_into_batches(negative_ids, negative_mask, bsize=bsize)


    batches = []
    for (q_ids, q_mask), (qexp_embs, qexp_wts), (p_ids, p_mask), (n_ids, n_mask), (p_scores,), (n_scores,) \
        in zip(query_batches, qexp_batches, positive_batches, negative_batches, positive_scores_batches, negative_scores_batches):

        # #!@ custom: previous for one negative sample
        # Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
        # Q_exp = (torch.cat((qexp_embs, qexp_embs)), torch.cat((qexp_wts, qexp_wts))) \
        #     if (qexp_embs is not None) else None
        # D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
        # batches.append((Q, D, Q_exp))

        #!@ custom: for many negative samples
        Q = (q_ids, q_mask)
        Q_exp = (qexp_embs, qexp_wts) if (qexp_embs is not None) else None 
        n_ids = n_ids.flatten(0,1)
        n_mask = n_mask.flatten(0,1)
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))

        if (p_scores[0] is not None):
            p_scores = torch.tensor(p_scores) # (N,)
            n_scores = torch.tensor(n_scores) # (N, n_negatives)
            pairwise_scores = (p_scores, n_scores)
        else:
            pairwise_scores = None
        batches.append((Q, D, Q_exp, pairwise_scores))

    return batches

def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices

#!@ original
# def _split_into_batches(ids, mask, bsize):
#     batches = []
#     for offset in range(0, ids.size(0), bsize):
#         batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

#     return batches

#!@ custom
def _split_into_batches(*inputs, bsize):
    batches = []
    for offset in range(0, len(inputs[0]), bsize):
        batches.append(tuple(x[offset:offset+bsize] for x in inputs))
    return batches

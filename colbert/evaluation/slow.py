import os

def slow_rerank(args, query, pids, passages):

    colbert = args.colbert
    inference = args.inference

    Q = inference.queryFromText([query])
    # Q: float tensor (1, query_maxlen, dim)

    D_ = inference.docFromText(passages, bsize=args.bsize)
    # D_: float tensor (1000, doc_maxlen, dim)

    scores = colbert.score(Q, D_).cpu()
    # scores: float tensor (1000)

    scores = scores.sort(descending=True)
    ranked = scores.indices.tolist()

    ranked_scores = scores.values.tolist()
    ranked_pids = [pids[position] for position in ranked]
    ranked_passages = [passages[position] for position in ranked]

    assert len(ranked_pids) == len(set(ranked_pids))

    return list(zip(ranked_scores, ranked_pids, ranked_passages))

from colbert.utils.utils import flatten

from colbert.indexing.loaders import get_parts, load_doclens

from colbert.labeling.index_ranker import IndexRankerRF
from colbert.ranking.index_part import IndexPart

class IndexPartRF(IndexPart):
    def __init__(self, directory, dim=128, part_range=None, verbose=True):
        first_part, last_part = (0, None) if part_range is None else (part_range.start, part_range.stop)

        # Load parts metadata
        all_parts, all_parts_paths, _ = get_parts(directory)
        self.parts = all_parts[first_part:last_part]
        self.parts_paths = all_parts_paths[first_part:last_part]

        # Load doclens metadata
        all_doclens = load_doclens(directory, flatten=False)

        self.doc_offset = sum([len(part_doclens) for part_doclens in all_doclens[:first_part]])
        self.doc_endpos = sum([len(part_doclens) for part_doclens in all_doclens[:last_part]])
        self.pids_range = range(self.doc_offset, self.doc_endpos)

        self.parts_doclens = all_doclens[first_part:last_part]
        self.doclens = flatten(self.parts_doclens)
        self.num_embeddings = sum(self.doclens)

        self.tensor = self._load_parts(dim, verbose)
        
        #!@ custom: ``IndexRankerRF`` is modified from ``IndexRaner``, to include query term weighting on expansion query terms
        self.ranker = IndexRankerRF(self.tensor, self.doclens)

    def rank(self, Q, pids):
        """
        Rank a single batch of Q x pids (e.g., 1k--10k pairs).
        """

        assert Q.size(0) in [1, len(pids)], (Q.size(0), len(pids))
        assert all(pid in self.pids_range for pid in pids), self.pids_range

        pids_ = [pid - self.doc_offset for pid in pids]
        scores = self.ranker.rank(Q, pids_)

        return scores
    
    def get(self, pids):
        """
        Load embeddings for the given pids
        """
        assert all(pid in self.pids_range for pid in pids), self.pids_range
        pids = [pid - self.doc_offset for pid in pids]
        return self.ranker.get(pids)

    #!@ custom: Add ``all_query_weights``
    def batch_rank(self, all_query_embeddings, query_indexes, pids, sorted_pids, all_query_weights):
        # all_query_embeddings: float tensor, size 3D (n_queries, dim, query_maxlen + exp_embs)
        # all_query_weights: float tensor, size 2D (n_queries, query_maxlen + exp_embs)
        """
        Rank a large, fairly dense set of query--passage pairs (e.g., 1M+ pairs).
        Higher overhead, much faster for large batches.
        """

        assert ((pids >= self.pids_range.start) & (pids < self.pids_range.stop)).sum() == pids.size(0)

        pids_ = pids - self.doc_offset
        scores = self.ranker.batch_rank(all_query_embeddings, query_indexes, pids_, sorted_pids, all_query_weights)

        return scores

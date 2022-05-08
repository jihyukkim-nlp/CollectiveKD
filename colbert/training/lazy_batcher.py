import torch
import os, numpy as np
import ujson
from tqdm import tqdm

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run


def load_collection(path):
    print_message("#> Loading collection...")

    collection = []

    with open(path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % (1000*1000) == 0:
                print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

            pid, passage = line.strip().split('\t')
            assert int(pid) == line_idx
            collection.append(passage)

    print()

    return collection

def load_static_supervision(path):
    print_message(f'#> Load static supervision from: {path}')

    static_supervision = {}

    with open(path) as f:
        
        for line_idx, line in enumerate(f):
            
            pos_score, neg_score, qid, pos_pid, neg_pid = line.strip().split('\t')
            
            pos_score, neg_score = map(float, (pos_score, neg_score))
            qid, pos_pid, neg_pid = map(int, (qid, pos_pid, neg_pid))

            static_supervision[qid] = static_supervision.get(qid, {})
            static_supervision[qid][pos_pid] = pos_score
            static_supervision[qid][neg_pid] = neg_score

    return static_supervision
    # return: Dict[qid -> Dict[pid -> score]]

def load_expansion_pt(path):
    print_message(f'#> Load {path}')
    exp_dict = torch.load(path) # Dict
    # > exp_dict['qid_to_embs']: Dict[ qid -> tensor (exp_embs, dim) ] = expansion emebddings
    # > exp_dict['qid_to_weights']: Dict[ qid -> tensor (exp_embs) ] = weights for the expansion embeddings (1 is used for original query token embeddings)
    # > exp_dict['qid_to_tokens']: Dict[ qid -> List[str] ] = list of expansion tokens (len=exp_embs)
    return exp_dict['qid_to_embs'], exp_dict['qid_to_weights']

class LazyBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = self._load_triples(args.triples, rank, nranks)
        self.queries = self._load_queries(args.queries)
        self.collection = load_collection(args.collection) 
        
        self.kd_query_expansion = args.kd_query_expansion
        if self.kd_query_expansion:
            self.qexp_embs_list, self.qexp_wts_list = [], [] 
            for kd_expansion_pt in args.kd_expansion_pt_list:
                qexp_embs, qexp_wts = load_expansion_pt(kd_expansion_pt)
                self.qexp_embs_list.append(qexp_embs)
                self.qexp_wts_list.append(qexp_wts)
        
        if args.static_supervision and os.path.exists(args.static_supervision):
            # Load precomputed pairwise relevance scores obtained by cross-encoder
            self.static_supervision = load_static_supervision(args.static_supervision)
            # self.static_supervision: Dict[qid -> Dict[pid -> score]]
        else:
            self.static_supervision = None


    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message("#> Loading triples...")

        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    qid, pos, *negs = ujson.loads(line)
                    triples.append((qid, pos, negs))

        return triples

    def _load_queries(self, path):
        print_message("#> Loading queries...")

        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query

        return queries

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        queries, positives, negatives = [], [], []

        for position in range(offset, endpos):
            
            qid, ppid, npids = self.triples[position]
            query, pos = self.queries[qid], self.collection[ppid]
            negs = [self.collection[npid] for npid in npids]

            if (self.static_supervision is not None):
                pos_score = self.static_supervision[qid][ppid]
                neg_scores = [self.static_supervision[qid][npid] for npid in npids]
                pos = (pos, pos_score)
                negs = list(zip(negs, neg_scores))
            else:
                pos = (pos, None)
                negs = list(zip(negs, [None]*len(negs)))

            if self.kd_query_expansion:

                qexp_embs_list, qexp_wts_list = [], [] 
                for qexp_embs, qexp_wts in zip(self.qexp_embs_list, self.qexp_wts_list):
                    qexp_embs_list.append(qexp_embs[qid]) # tensor (exp_embs, dim)
                    qexp_wts_list.append(qexp_wts[qid]) # tensor (exp_embs)
                query = (query, torch.stack(qexp_embs_list, dim=0), torch.stack(qexp_wts_list, dim=0))
            else:
                query = (query, None, None) # ``None`` for compatibility

            queries.append(query)
            positives.append(pos)
            negatives.append(negs)
        
        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        # queries: List[ Tuple(str, List[int] (or None), List[float] (or None) ] = list of tuples of 1) query, 2) list of token positions on the relevant passage used for expansion, and 3) list of term weight for fb_embs tokens.
        # positives, negatives: List[ Tuple(str, List[int]) ] = for each query (outer list), tuple of ([positive/negative] passage, list of token positions used for matching)
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx

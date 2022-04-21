from argparse import Namespace
from functools import partial
import numpy as np
import os
from tqdm import tqdm
import faiss
import time

from multiprocessing import Pool

import torch

from colbert.modeling.inference import ModelInference
from colbert.parameters import DEVICE
from colbert.utils.utils import print_message

import threading
import queue
from colbert.indexing.loaders import get_parts

from sklearn.cluster import KMeans
# class FaissKMeans:
#     def __init__(self, n_clusters=10, n_init=10, max_iter=300):
#         self.n_clusters = n_clusters
#         self.n_init = n_init
#         self.max_iter = max_iter
#         self.kmeans = None
#         self.cluster_centers_ = None
#         self.inertia_ = None

#     def fit(self, X):
#         self.kmeans = faiss.Kmeans(d=X.shape[1],
#                                    k=self.n_clusters,
#                                    niter=self.max_iter,
#                                    nredo=self.n_init)
#         self.kmeans.train(X.astype(np.float32))
#         self.cluster_centers_ = self.kmeans.centroids
#         self.inertia_ = self.kmeans.obj[-1]

#     # def predict(self, X):
#     #     return self.kmeans.index.search(X.astype(np.float32), 1)[1]

    
from colbert.labeling.faiss_index import FaissIndex
from colbert.labeling.index_part import IndexPartRF
class ColbertPRF():
    
    def __init__(self, args, inference: ModelInference, faiss_depth=1024,):

        if args.fb_k > 0 and args.beta > 0.0:

            self.inference = inference
            self.faiss_depth = faiss_depth

            # For ANN Search
            self.faiss_index = FaissIndex(args.index_path, args.faiss_index_path, args.nprobe,
                                            part_range=args.part_range, inference=inference)
            if faiss_depth is not None:
                self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

            self.index = IndexPartRF(args.index_path, dim=inference.colbert.dim, part_range=args.part_range, verbose=True)

            # New code for PRF
            self.fb_docs = args.fb_docs # number of docs for kmeans clustering
            self.fb_clusters = args.fb_clusters # number of clusters (centroids)
            self.fb_k = args.fb_k # number of expansion embeddings to add to the query
            self.beta = args.beta # weight factor for the expansion embeddings
            self.kmeans_init = args.kmeans_init # initialization method for KMeans clustering
            print_message(f'ColBertPRF config: fb_docs={self.fb_docs}, fb_clusters={self.fb_clusters}, fb_k={self.fb_k}, beta={self.beta}, kmeans_init={self.kmeans_init}')

            print_message("Computing IDF")
            self.skips = set(self.inference.query_tokenizer.tok.special_tokens_map.values())
            num_docs = self.faiss_index.num_docs
            self.idfdict = {}
            for tid in tqdm(range(self.inference.query_tokenizer.tok.vocab_size)):
                df = self.getDF_by_id(tid)
                idfscore = np.log((1.0 + num_docs) / (df + 1))
                self.idfdict[tid] = idfscore
            self.get_nn = partial(self.faiss_index.queries_to_embedding_ids, 10)
        
        else:
            print_message(f'ColBertPRF config: DO NOT perform query expansion')
            self.prepare_ranges(args.index_path, args.dim, args.step, args.part_range)

    def prepare_ranges(self, index_path, dim, step, part_range):
        print_message("#> Launching a separate thread to load index parts asynchronously.")
        parts, _, _ = get_parts(index_path)

        positions = [(offset, offset + step) for offset in range(0, len(parts), step)]

        if part_range is not None:
            positions = positions[part_range.start: part_range.stop]

        loaded_parts = queue.Queue(maxsize=2)

        def _loader_thread(index_path, dim, positions):
            for offset, endpos in positions:
                #!@ custom: for query term weighting
                index = IndexPartRF(index_path, dim=dim, part_range=range(offset, endpos), verbose=True)
                loaded_parts.put(index, block=True)

        thread = threading.Thread(target=_loader_thread, args=(index_path, dim, positions,))
        thread.start()

        self.positions = positions
        self.loaded_parts = loaded_parts
        self.thread = thread
    
    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)
        Q = self.inference.queryFromText(queries, bsize=512 if len(queries) > 512 else None)
        return Q

    def get_nearest_tokens_for_embs(self, embs, low_tf=0):
        """
            Returns the most related terms for each of a number of given embeddings
        """
        from collections import defaultdict
        assert len(embs.shape) == 2
        n_centroid, dim = embs.shape

        embs = torch.tensor(embs).unsqueeze(0).to(DEVICE)

        ids = self.get_nn(embs, verbose=False)#[0] #(1 x 240)
        ids = ids.view(n_centroid, 10)

        rtrs = []
        for id_set in ids:
            id2freq = defaultdict(int)
            for id in id_set:
                id2freq[self.faiss_index.emb2tid[id].item()] += 1
            rtr = {}
            for t, freq in sorted(id2freq.items(), key=lambda item: -1 * item[1]):
                if freq <= low_tf:
                    continue
                token = self.inference.query_tokenizer.tok.decode([t])
                if "[unused" in token or token in self.skips:
                    continue
                rtr[token] = freq
            rtrs.append(rtr)
        return rtrs
    
    def getDF_by_id(self, tid):
        """
            Returns the document frequency of a given token id
        """
        return self.faiss_index.dfs[tid].item()

    #!@ custom: main function: (query token embeddings for a query, pids for expansion) -> (exp_embs, exp_weights, exp_tokens)
    def expand(self, q_embs, fb_pids=None):
        # q_embs: float32 tensor, (query_maxlen, dim)
        q_embs = q_embs.unsqueeze(0)

        # Get embeddings for feedback documents
        if fb_pids is not None:
            assert type(fb_pids) in [list, tuple], type(fb_pids)
            assert all(type(pid) is int for pid in fb_pids)

            # feedback documents as top-``fb_docs`` ranked documents
            fb_embs = self.index.get(fb_pids)
            # List[ 2d tensor ]
        
        else:
            #TODO: we need IndexRanker
            raise NotImplementedError
            # Retrieve candidates for feedback documents
            fb_pids = self.retrieve(q_embs, verbose=False)[0]
            # time for retrieve: 0.03 ~ 0.06 seconds

        # concatenate all token embeddings in the feedback documents
        fb_embs_concat = torch.cat(fb_embs, dim=0)
        # fb_embs_concat = fb_embs.view(-1, fb_embs.size(2)).contiguous()

        n_clusters = min(self.fb_clusters, len(fb_embs_concat))

        # Prepare initial centroid embeddings for effective, efficient K-means clustering
        if self.kmeans_init=='avg_step_position':
            #?@ option 1: init from all fb_embs
            if len(fb_embs) > 1:
                _cluster_indices = np.linspace(0, n_clusters, len(fb_embs)+1).astype(np.int64)
                sts, eds = _cluster_indices[:-1], _cluster_indices[1:]
                centroids_init = np.zeros((n_clusters, self.inference.colbert.dim), dtype=np.float32)
                for i, (st, ed) in enumerate(zip(sts, eds)):
                    _embs = fb_embs[i] 
                    _step_positions = np.linspace(0, len(_embs)-1, ed - st).astype(np.int64)
                    centroids_init[st:ed] = _embs[_step_positions, :].cpu().data.numpy()
            else:
                _step_positions = np.linspace(0, len(fb_embs_concat)-1, n_clusters).astype(np.int64)
                centroids_init = fb_embs_concat[_step_positions, :].cpu().data.numpy()
        
        elif self.kmeans_init=='top1_step_position':
            #?@ option 2: init from the top-1 ranked fb_embs
            _top1_fb_embs = fb_embs[0]
            _step_positions = np.linspace(0, len(_top1_fb_embs)-1, n_clusters).astype(np.int64)
            centroids_init = _top1_fb_embs[_step_positions, :].cpu().data.numpy()
        
        elif self.kmeans_init=='random':
            centroids_init = None

        # Perform K-means clustering
        if (centroids_init is not None):
            kmn = KMeans(n_clusters, init=centroids_init, n_init=1)
        else:
            kmn = KMeans(n_clusters)
        kmn.fit(fb_embs_concat.cpu().numpy())
        # kmn = FaissKMeans(n_clusters=n_clusters)
        # kmn.fit(fb_embs_concat)
        centroids = np.float32(kmn.cluster_centers_)

        # Search nearest neighbor tokens to the centroids, from the tokens in the entire collection
        toks2freqs = self.get_nearest_tokens_for_embs(centroids)
        # time for get_nearest_tokens_for_embs: 0.02 seconds (for 24 centroids)
        # List of dicts (token : Document frequency)

        # Rank the clusters by descending idf
        triples = [] # list of (exp_emb, exp_weight, exp_token)
        for cluster_idx, tok2freq in zip(range(n_clusters), toks2freqs):
            if len(tok2freq) == 0:
                continue
            most_likely_tok = max(tok2freq, key=tok2freq.get)
            tid = self.inference.query_tokenizer.tok.convert_tokens_to_ids(most_likely_tok)
            
            exp_emb = centroids[cluster_idx]
            exp_wt = self.idfdict[tid] * self.beta
            exp_tok = most_likely_tok
            triples.append((exp_emb, exp_wt, exp_tok))

        topk_triples = sorted(triples, key=lambda tup: -tup[1])[:self.fb_k]
        exp_embs, exp_weights, exp_tokens = zip(*topk_triples)

        # Post-processing: data formatting
        exp_embs = torch.tensor(exp_embs, dtype=q_embs.dtype, device=q_embs.device)
        exp_weights = torch.tensor(exp_weights, dtype=q_embs.dtype, device=q_embs.device)
        exp_tokens = list(exp_tokens)

        return exp_embs, exp_weights, exp_tokens

    def expandFromText(self, query, fb_pids=None):
        # query: str = the given query 
                
        # Encode query
        q_embs = self.inference.queryFromText(queries=[query]).squeeze(0)
        # q_embs: cpu, float-32 tensor (1, query_maxlen, dim) = query token embeddings
        
        exp_embs, exp_weights, exp_tokens = self.expand(q_embs, fb_pids=fb_pids)
        return q_embs, exp_embs, exp_weights, exp_tokens




if __name__=='__main__':
    """
awk '{ print $1 }' /hdd/jihyuk/DataCenter/MSMARCO/qrels.dev.small.tsv | uniq -c | sort -nr |head
    4 810394
    4 592235
    4 565696
    4 504335
    4 457622
    4 456551
    4 174592
    4 1084838
    3 899800
    3 887398
    
cat /hdd/jihyuk/DataCenter/MSMARCO/qrels.dev.small.tsv | grep -w "810394"
    810394  0       7966804 1
    810394  0       764872  1
    810394  0       7966805 1
    810394  0       7966808 1

cat /hdd/jihyuk/DataCenter/MSMARCO/queries.dev.small.tsv | grep -w "810394"
    810394  what is the cause of the symptoms nausea throwing up

sed -n "7966805p" /hdd/jihyuk/DataCenter/MSMARCO/collection.tsv 
    7966804 Viral gastroenteritis Gastroenteritis (stomach flu) is a viral condition that causes diarrhea and vomiting. Giardiasis Giardiasis is an infection of the small intestine causing diarrhea, gas, bloating, nausea and stomach cramps.
sed -n "764873p" /hdd/jihyuk/DataCenter/MSMARCO/collection.tsv 
    764872  Nausea and vomiting may also occur when there are metabolic changes in the body, such as during early pregnancy, or when people have diabetes that is severely out of control or severe liver failure or kidney failure. Psychologic problems also can cause nausea and vomiting (known as functional or psychogenic vomiting).
sed -n "7966806p" /hdd/jihyuk/DataCenter/MSMARCO/collection.tsv 
    7966805 In such disorders (for example, appendicitis or pancreatitis), it is typically the pain rather than the vomiting that causes people to seek medical care. Many drugs, including alcohol, opioid analgesics (such as morphine), and chemotherapy drugs, can cause nausea and vomiting.
sed -n "7966809p" /hdd/jihyuk/DataCenter/MSMARCO/collection.tsv 
    7966808 Nausea is a sensation of unease and discomfort in the upper stomach with an involuntary urge to vomit. It may precede vomiting, but a person can have nausea without vomiting. When prolonged, it is a debilitating symptom. Nausea is a non-specific symptom, which means that it has many possible causes. Some common causes of nausea are motion sickness, dizziness, migraine, fainting, low blood sugar, gastroenteritis (stomach infection) or food poisoning.


exp_thr=0.5, exp_embs=10, exp_beta=0.5, exp_mmr_thr=0.9
exp_embs (10, 128) (torch.float32)
exp_weights (10,) (torch.float32) = 
        [0.468, 0.457, 0.451, 0.45, 0.446, 0.446, 0.438, 0.433, 0.431, 0.428]
exp_tokens (10) = 
        ['nausea', '[CLS]', '[SEP]', 'causes', '[SEP]', 'and', 'causes', 'as', 'vomiting', 'or']
expand_a_query_using_colbert_rf... elapsed: 0.024 sec (real) / 0.180 sec (cpu)                                                                                                     
    """
    
    
    query = "what is the cause of the symptoms nausea throwing up"
    relevant_passages = [
        "Viral gastroenteritis Gastroenteritis (stomach flu) is a viral condition that causes diarrhea and vomiting. Giardiasis Giardiasis is an infection of the small intestine causing diarrhea, gas, bloating, nausea and stomach cramps.",
        "Nausea and vomiting may also occur when there are metabolic changes in the body, such as during early pregnancy, or when people have diabetes that is severely out of control or severe liver failure or kidney failure. Psychologic problems also can cause nausea and vomiting (known as functional or psychogenic vomiting).",
        "In such disorders (for example, appendicitis or pancreatitis), it is typically the pain rather than the vomiting that causes people to seek medical care. Many drugs, including alcohol, opioid analgesics (such as morphine), and chemotherapy drugs, can cause nausea and vomiting.",
        "Nausea is a sensation of unease and discomfort in the upper stomach with an involuntary urge to vomit. It may precede vomiting, but a person can have nausea without vomiting. When prolonged, it is a debilitating symptom. Nausea is a non-specific symptom, which means that it has many possible causes. Some common causes of nausea are motion sickness, dizziness, migraine, fainting, low blood sugar, gastroenteritis (stomach infection) or food poisoning.",
    ]

    from types import SimpleNamespace
    args = SimpleNamespace()

    args.index_path = 'experiments/colbert.teacher/MSMARCO-psg/index.py/MSMARCO.L2.32x200k/'
    args.faiss_index_path = 'experiments/colbert.teacher/MSMARCO-psg/index.py/MSMARCO.L2.32x200k/ivfpq.32768.faiss'
    args.nprobe = 32
    args.part_range = None

    # args.collection = '/workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv'
    # from colbert.training.lazy_batcher import load_collection
    # args.collection = load_collection(args.collection)

    args.query_maxlen = 32
    args.doc_maxlen = 180
    args.dim = 128
    args.similarity = 'l2'
    args.mask_punctuation = True
    args.checkpoint = 'data/checkpoints/colbert.teacher.dnn'
    from colbert.evaluation.load_model import load_model
    inference = ModelInference(colbert=load_model(args)[0])

    args.fb_docs=3
    args.fb_clusters=24
    args.fb_k=10
    args.beta=0.5
    colbert_prf = ColbertPRF(args=args, inference=inference, faiss_depth=1024)

    from pilot_test.time_consumption.utils import elapsed
    @elapsed
    def test_execute():
        # q_embs = inference.queryFromText(queries=[query])

        # return colbert_prf.expand(q_embs=q_embs, fb_pids=[7966804, 764872, 7966805])
        return colbert_prf.expandFromText(query, fb_pids=[0,1,2]) #?@ debugging
        # return colbert_prf.expandFromText(query, fb_pids=[7966804, 764872, 7966805, 7966808])
        # return colbert_prf.expandFromText(query, fb_pids=[7966804, 764872, 7966805])
        # return colbert_prf.expandFromText(query, fb_pids=None)
    
    for _ in range(10):
        q_embs, exp_embs, exp_weights, exp_tokens = test_execute()
    
    #?@ debugging
    print(f'fb_docs={args.fb_docs}, fb_clusters={args.fb_clusters}, fb_k={args.fb_k}, beta={args.beta}')
    print(f'query: {query}')
    print(f'q_embs {tuple(q_embs.size())} ({q_embs.dtype})')
    print(f'exp_embs {tuple(exp_embs.size())} ({exp_embs.dtype})')
    print(f'exp_weights {tuple(exp_weights.size())} ({exp_weights.dtype}) = \n\t{[float(str(x)[:5]) for x in exp_weights.tolist()]}')
    print(f'exp_tokens ({len(exp_tokens)}) = \n\t{exp_tokens}')
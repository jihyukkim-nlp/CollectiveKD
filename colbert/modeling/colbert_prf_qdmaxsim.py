from argparse import Namespace
import torch

from colbert.modeling.inference import ModelInference
from colbert.parameters import DEVICE

import threading
import queue
from colbert.utils.utils import print_message
from colbert.indexing.loaders import get_parts
from colbert.labeling.index_part import IndexPartRF

class ColbertPrfQDMaxsim():
    
    def __init__(self, args, inference: ModelInference):
        self.inference = inference

        self.fb_k = args.fb_k # number of expansion embeddings to add to the query
        self.beta = args.beta # weight factor for the expansion embeddings

        self.collection = args.collection

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
    
    def get_masked_document_tokens(self, docs):
        # docs: List[str] = list of relevant documents to the given query

        # Tokenize docs
        docs_tokens_all = self.inference.doc_tokenizer.tokenize(batch_text=docs, add_special_tokens=True)

        # Mask tokens
        _skiplist = self.inference.colbert.skiplist
        docs_tokens_masked = [[t for t in tokens if t not in _skiplist and t!='[PAD]'] for tokens in docs_tokens_all]
        # docs_tokens_masked: List[ List[str] ] = list of (list of tokens) in relevant documents

        return docs_tokens_masked

    def sort_by_weight(self, weight):
        sorted_weight, sorted_d_idx = torch.sort(weight, descending=True)

        # Drop irrelevant doc tokens to the query
        _mask = sorted_weight > 0.0
        sorted_weight = sorted_weight[_mask]
        sorted_d_idx = sorted_d_idx[_mask]

        return sorted_weight.tolist(), sorted_d_idx.tolist()

    def expansion_term_selection(self, q_embs, d_embs, d_toks):
        # q_embs: cpu, float-32 tensor (query_maxlen, dim) = query token embeddings
        # d_embs: cpu, float-32 tensor (n_docs * doc_maxlen, dim) = concatenated doc token embeddings for relevant documents
        
        # Compute cosine similarity
        with torch.no_grad():
            # Q-D cosine similarity
            similarity = (q_embs @ d_embs.transpose(0,1))
            # similarity: cpu, float-32 tensor (query_maxlen, n_docs * doc_maxlen) = cosine similarity between query/document tokens
    
            # d_toks_scores
            similarity = similarity.transpose(0,1)
            # similarity: cpu, float-32 tensor (n_docs * doc_maxlen, query_maxlen)
            closest_q_idx = torch.argmax(similarity, dim=1)
            # closest_q_idx: cpu, int-64 tensor (n_docs * doc_maxlen) = the most similar query token index for each doc token
            d_toks_scores = similarity[torch.arange(closest_q_idx.shape[0]), closest_q_idx]
            # d_toks_scores: cpu, float-32 tensor (n_docs * doc_maxlen) = maximum cosine similarity of doc tokens to the given query

        sorted_weight, sorted_index = self.sort_by_weight(d_toks_scores)
        # sorted_weight : List[float] = list of sorted cosine similarity
        # sorted_index  : List[int  ] = list of sorted doc token indices

        # Obtain expansion embeddings, as top-``fb_k`` doc token embeddings according to similarities,
        exp_embs = torch.stack([d_embs[_] for _ in sorted_index[:self.fb_k]], dim=0)
        exp_tokens = [d_toks[_] for _ in sorted_index[:self.fb_k]]
        exp_weights = torch.tensor(sorted_weight[:self.fb_k], device=exp_embs.device, dtype=exp_embs.dtype)
        # exp_embs: cpu, float-32 tensor (exp_embs, dim) = expansion embeddings
        # exp_weights: cpu, float-32 tensor (exp_embs) = weights for the expansion embeddings
        # exp_tokens: List[str] = list of expansion tokens (len=exp_embs)
        
        return exp_embs, exp_weights, exp_tokens

    def expand(self, q_embs, fb_pids):
        # q_embs: float32 tensor, (query_maxlen, dim)
        # fb_pids: List[int] = list of feedback documents

        # Encode feedback documents
        docs = [self.collection[pid] for pid in fb_pids]
        d_embs = self.inference.docFromText(docs, bsize=512, keep_dims=False, float16=False)
        # d_embs: List[cpu, float-16 tensor (doc_maxlen, dim)] = list of pseudo-query embeddings in documents in the collection.
        # (tokens included in ``inference.colbert.skiplist`` are excluded)
        d_toks = self.get_masked_document_tokens(docs)
        # d_toks: List[str] = list of document tokens (tokens included in ``inference.colbert.skiplist`` are excluded)

        # Concatenate docs
        d_embs = torch.cat(d_embs, dim=0)
        d_toks = [t for toks in d_toks for t in toks]
        # d_embs: cpu, float-32 tensor (n_docs * doc_maxlen, dim) = concatenated doc token embeddings for relevant documents
        # d_toks: List[str] = list of concatenated document tokens
        # (tokens included in ``self.inference.colbert.skiplist`` are excluded)

        exp_embs, exp_weights, exp_tokens = self.expansion_term_selection(q_embs=q_embs, d_embs=d_embs, d_toks=d_toks)
        assert exp_embs.size(0) == exp_weights.size(0) == len(exp_tokens)
        # exp_embs: cpu, float-32 tensor (exp_embs, dim) = expansion embeddings
        # exp_weights: cpu, float-32 tensor (exp_embs) = weights for the expansion embeddings
        # exp_tokens: List[str] = list of expansion tokens (len=exp_embs)

        # Multiply ``beta`` to ``exp_weights``
        exp_weights = self.beta * exp_weights

        return exp_embs, exp_weights, exp_tokens
    

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
    args.query_maxlen = 32
    args.doc_maxlen = 180
    args.dim = 128
    args.similarity = 'l2'
    args.mask_punctuation = True
    args.checkpoint = 'data/checkpoints/colbert.teacher.dnn'
    from colbert.evaluation.load_model import load_model
    inference = ModelInference(colbert=load_model(args)[0])

    args.exp_thr=0.0
    args.exp_embs=10
    args.exp_beta=1.0
    args.exp_mmr_thr=0.9
    colbert_rf = ColbertRF(inference=inference, 
                            exp_thr=args.exp_thr, exp_embs=args.exp_embs, exp_beta=args.exp_beta, exp_mmr_thr=args.exp_mmr_thr)

    from pilot_test.time_consumption.utils import elapsed
    @elapsed
    def expand_a_query_using_colbert_rf():
        return colbert_rf.expandFromText(query, relevant_passages)
    
    for _ in range(10):
        q_embs, exp_embs, exp_weights, exp_tokens = expand_a_query_using_colbert_rf()
    
    #?@ debugging
    print(f'exp_thr={args.exp_thr}, exp_embs={args.exp_embs}, exp_beta={args.exp_beta}, exp_mmr_thr={args.exp_mmr_thr}')
    print(f'exp_embs {tuple(exp_embs.size())} ({exp_embs.dtype})')
    print(f'exp_weights {tuple(exp_weights.size())} ({exp_weights.dtype}) = \n\t{[float(str(x)[:5]) for x in exp_weights.tolist()]}')
    print(f'exp_tokens ({len(exp_tokens)}) = \n\t{exp_tokens}')

import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE


class ColBERT(BertPreTrainedModel):
    def __init__(self, config,
        query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}
            
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    #!@ custom
    def forward(self, Q, D, Q_exp=None):
        return self.score(self.query(*Q), self.doc(*D), Q_exp=Q_exp)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True, float16=True):
        
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        
        D = self.bert(input_ids, attention_mask=attention_mask)[0] 
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        
        D = D * mask
        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            #!@ original
            # D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            # D = [d[mask[idx]] for idx, d in enumerate(D)]

            #!@ custom
            mask = mask.cpu().bool().squeeze(-1)
            D = D.cpu()
            if float16: 
                D = D.to(dtype=torch.float16)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D, Q_exp=None):
        # Q       : (bsize        , query_maxlen  , dim) = e.g., bsize=2 -> [Q1, Q2]
        # D       : ((k+1) * bsize, doc_maxlen    , dim) = e.g., bsize=2 -> [P1, P2, N11, ..., N1k, N21, ..., N2k]
        # Q_exp[0]: (bsize        , n_exp_embs    , dim)
        # Q_exp[1]: (bsize        , n_exp_embs)

        Q_list = []
        Q_wts_list = []
        if (Q_exp is not None):
            exp_embs_multi, exp_wts_multi = Q_exp
            for exp_embs, exp_wts in zip(exp_embs_multi.transpose(0,1), exp_wts_multi.transpose(0,1)):

                exp_embs, exp_wts = exp_embs.to(DEVICE), exp_wts.to(DEVICE)

                #?@ debugging
                # print(f'\nQ_exp=\n\t{Q_exp} ({len(Q_exp)})')
                # print(f'exp_embs=\n\t{exp_embs} ({exp_embs.shape})')
                # print(f'exp_wts=\n\t{exp_wts} ({exp_wts.shape})')
                
                ones_wts = torch.ones(Q.size(0), Q.size(1), dtype=exp_wts.dtype, device=DEVICE)
                Q_wts = torch.cat((ones_wts, exp_wts), dim=1) # (``bsize, query_maxlen + n_exp_embs``)

                Q = torch.cat((Q, exp_embs), dim=1) # (``bsize, query_maxlen + n_exp_embs, dim``)
                
                Q_list.append(Q)
                Q_wts_list.append(Q_wts)

            #?@ debugging
            # exit()

        else:
            Q_list.append(Q)
            Q_wts_list.append(None)

        if self.similarity_metric == 'cosine': # for validation
            
            if self.training:
                raise NotImplementedError

            relevance_list = []
            for Q, Q_wts in zip(Q_list, Q_wts_list):
                if Q.size(0) > 1:
                    n_negatives = (D.size(0)//Q.size(0)) - 1
                    Q_repeat = Q.unsqueeze(1).repeat(1, n_negatives, 1, 1) # (``bsize, k, query_maxlen, dim``): [[Q1, ..., Q1], [Q2, ...., Q2]]
                    Q_repeat = Q_repeat.flatten(0, 1) # (``bsize * k, query_maxlen, dim``): [Q1, Q1, ..., Q1, Q2, Q2, ..., Q2]
                    Q = torch.cat((Q, Q_repeat), dim=0) # (``(k+1) * bsize, query_maxlen, dim``): [Q1, Q2, Q1, Q1, ..., Q1, Q2, Q2, ..., Q2]

                QD = (Q @ D.permute(0, 2, 1)) # (``(k+1) * bsize, query_maxlen, doc_maxlen``)
                maxsim = QD.max(2).values # (``(k+1) * bsize, query_maxlen``)
                if (Q_wts is not None):
                    if Q.size(0) > 1:
                        Q_wts_repeat = Q_wts.unsqueeze(1).repeat(1, n_negatives, 1) # (``bsize, k, query_maxlen``): [[Q1, ..., Q1], [Q2, ...., Q2]]
                        Q_wts_repeat = Q_wts_repeat.flatten(0, 1) # (``bsize * k, query_maxlen``): [Q1, Q1, ..., Q1, Q2, Q2, ..., Q2]
                        Q_wts = torch.cat((Q_wts, Q_wts_repeat), dim=0) # (``(k+1) * bsize, query_maxlen``): [Q1, Q2, Q1, Q1, ..., Q1, Q2, Q2, ..., Q2]
                    maxsim = maxsim * Q_wts # (``(k+1) * bsize, query_maxlen``)
                relevance = maxsim.sum(1) # (``(k+1) * bsize``)
                relevance_list.append(relevance)
            return torch.stack(relevance_list, dim=0).mean(dim=0)

        assert self.similarity_metric == 'l2'
        
        if Q_list[0].size(0)>1: # during training
            
            relevance_list = []
            for _Q, _Q_wts in zip(Q_list, Q_wts_list):

                #?@ debugging
                # print(f'_Q ({_Q.shape})')
                # print(f'_Q_wts ({_Q_wts.shape})')

                _Q = _Q.unsqueeze(2).unsqueeze(1) 
                # (bsize, 1          , query_maxlen, 1         , dim)
                _D = D.unsqueeze(1).unsqueeze(0) 
                # (1    , (k+1) * bsize, 1           , doc_maxlen, dim)
                QD = -1.0 * (( _Q - _D )**2).sum(-1)
                # (bsize, (k+1) * bsize, query_maxlen, doc_maxlen)
                maxsim = QD.max(-1).values
                # (bsize, (k+1) * bsize, query_maxlen)
                if (_Q_wts is not None): # for teacher
                    _Q_wts = _Q_wts.unsqueeze(1) 
                    # (bsize, 1,         query_maxlen)
                    maxsim = maxsim * _Q_wts
                relevance = maxsim.sum(-1) # (bsize, (k+1) * bsize)
                
                #?@ debugging
                # print(f'relevance={relevance.shape}')
                
                relevance_list.append(relevance)
            
            # if len(relevance_list)==1: # student
            #     return relevance_list[0]
            # else:
            #     relevance_list
            # return torch.stack(relevance_list, dim=0).mean(0)
            # assert self.training
            return relevance_list
            # if (Q_wts_list[0] is None):
            #     assert len(relevance_list)==1, f'Teacher must take expansion embeddings.'
            #     return relevance_list[0]
            # else:
            #     return relevance_list
        
        else: # during validation: a single Q for multiple D
            
            assert len(Q_list)==1
            
            Q = Q_list[0]
            
            # Q: (1          , query_maxlen  , dim)
            # D: (n_candiates, doc_maxlen    , dim)

            # Used for re-rank evaluation (refer to `colbert/evaluation/slow.py: slow_rerank`)
            QD = -1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1) # (n_candiates, query_maxlen, doc_maxlen)
            maxsim = QD.max(-1).values # (n_candiates, query_maxlen)
            if (Q_wts is not None):
                raise Exception("We do not use query term weighting for evaluation")
                maxsim = maxsim * Q_wts # (n_candiates, query_maxlen)
            relevance = maxsim.sum(-1) # (n_candiates)
        
            return relevance



    # #!@ custom
    # def forward(self, Q, D, Q_exp=None, inbatch_negatives=False):
    #     return self.score(self.query(*Q), self.doc(*D), Q_exp=Q_exp, inbatch_negatives=inbatch_negatives)

    # def score(self, Q, D, Q_exp=None, inbatch_negatives=False):
    #     # Q     : (2 * bsize, query_maxlen, dim) = e.g., bsize=3 -> [Q1, Q2, Q3, Q1, Q2, Q3]
    #     # D     : (2 * bsize, doc_maxlen, dim)   = e.g., bsize=3 -> [P1, P2, P3, N1, N2, N3]
    #     # Q_exp : (2 * bsize, n_exp_embs, dim), (2 * bsize, n_exp_embs)

    #     if (Q_exp is not None):
    #         exp_embs, exp_wts = Q_exp
    #         exp_embs, exp_wts = exp_embs.to(DEVICE), exp_wts.to(DEVICE)

    #         ones_wts = torch.ones(Q.size(0), Q.size(1), dtype=exp_wts.dtype, device=DEVICE)
    #         Q_wts = torch.cat((ones_wts, exp_wts), dim=1) # (2 * bsize, query_maxlen + n_exp_embs)

    #         Q = torch.cat((Q, exp_embs), dim=1) # (2 * bsize, query_maxlen + n_exp_embs, dim)
    #     else:
    #         Q_wts = None

    #     if self.similarity_metric == 'cosine':
    #         QD = (Q @ D.permute(0, 2, 1)) # (2 * bsize, query_maxlen, doc_maxlen)
    #         maxsim = QD.max(2).values # (2 * bsize, query_maxlen)
    #         if (Q_wts is not None):
    #             maxsim = maxsim * Q_wts 
    #         relevance = maxsim.sum(1) # (2 * bsize)
    #         return relevance

    #     assert self.similarity_metric == 'l2'
        
    #     if inbatch_negatives:
    #         _bsize = Q.shape[0]
    #         #!@ custom: in-batch negatives
    #         Q = Q[:_bsize // 2].unsqueeze(2).unsqueeze(1) 
    #         # (bsize,   1,          query_maxlen,   1,          dim)
    #         D = D.unsqueeze(1).unsqueeze(0) 
    #         # (1,       2 * bsize,  1,              doc_maxlen, dim)
    #         QD = -1.0 * (( Q - D )**2).sum(-1)
    #         # (bsize, 2 * bsize, query_maxlen, doc_maxlen)
    #         maxsim = QD.max(-1).values
    #         # (bsize, 2 * bsize, query_maxlen)
    #         if (Q_wts is not None):
    #             Q_wts = Q_wts[:_bsize // 2].unsqueeze(1) 
    #             # (bsize, 1,         query_maxlen)
    #             maxsim = maxsim * Q_wts
    #         relevance = maxsim.sum(-1) # (bsize, 2 * bsize)
    #         return relevance
    #         # return (-1.0 * ((Q[:Q.shape[0] // 2].unsqueeze(2).unsqueeze(1) - D.unsqueeze(1).unsqueeze(0))**2).sum(-1)).max(-1).values.sum(-1)
        
    #     #!@ original: pair-wise negatives
    #     # Used for re-rank evaluation (refer to `colbert/evaluation/slow.py: slow_rerank`)
    #     QD = -1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1) # (2 * bsize, query_maxlen, doc_maxlen)
    #     maxsim = QD.max(-1).values # (2 * bsize, query_maxlen)
    #     if (Q_wts is not None):
    #         maxsim = maxsim * Q_wts # (2 * bsize, query_maxlen)
    #     relevance = maxsim.sum(-1) # (2 * bsize)
    #     return relevance

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

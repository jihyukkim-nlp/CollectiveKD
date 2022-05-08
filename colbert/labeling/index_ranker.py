import os
import math
import torch
import ujson
import traceback

from itertools import accumulate
from colbert.parameters import DEVICE
from colbert.utils.utils import print_message, dotdict, flatten

BSIZE = 1 << 14

from colbert.ranking.index_ranker import IndexRanker, torch_percentile
class IndexRankerRF(IndexRanker):
    def __init__(self, tensor, doclens):
        self.tensor = tensor
        self.doclens = doclens

        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens))

        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)

        self.dim = self.tensor.size(-1)

        self.strides = [torch_percentile(self.doclens, p) for p in [90]]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides)))

        print_message(f"#> Using strides {self.strides}..")

        self.views = self._create_views(self.tensor)

        nranks = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])
        nranks = max(1, nranks)
        is_distributed = nranks > 1
        if is_distributed:
            self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, set(['cpu']+[f'cuda:{rank}' for rank in range(nranks)]))
        else:
            self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, {'cpu', 'cuda:0'})
        
    #!@ custom: Add ``Q_weight``
    def rank(self, Q, pids, views=None, shift=0, Q_weight=None):
        # Q: float tensor, (bsize, dim, query_maxlen + exp_embs)
        # Q_weight: float tensor, (bsize, query_maxlen + exp_embs)
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]

        Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype) # cuda, float32
        Q_weight = Q_weight.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype) # cuda, float32

        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], []

        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)
            
            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator] # (bsize, dim,  query_maxlen + exp_embs)
            group_Q_weight = Q_weight if Q.size(0)==1 else Q_weight[locator] # (bsize, query_maxlen + exp_embs)

            group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
            D = D.to(DEVICE)
            D = D[group_offsets_expand.to(DEVICE)].to(dtype=self.maxsim_dtype)

            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)

            # scores = (D @ group_Q) * mask.unsqueeze(-1)
            # scores = scores.max(1).values.sum(-1).cpu()
            QD = (D @ group_Q) * mask.unsqueeze(-1) # (bsize, doc_maxlen, query_maxlen + exp_embs)
            maxsim = QD.max(1).values # (bsize, query_maxlen + exp_embs)
            scores = (maxsim * group_Q_weight).sum(-1).cpu() # (bsize)

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation].tolist()

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        return output_scores

    #!@ custom: Add ``all_query_weights``
    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids, all_query_weights):
        # all_query_embeddings: float tensor, size 3D (n_queries, dim, query_maxlen + exp_embs)
        # all_query_weights: float tensor, size 2D (n_queries, query_maxlen + exp_embs)

        assert sorted_pids is True

        ######

        scores = []
        range_start, range_end = 0, 0

        for pid_offset in range(0, len(self.doclens), 50_000):
            pid_endpos = min(pid_offset + 50_000, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            print_message(f"###--> Got {len(pids)} query--passage pairs in this sub-range {(pid_offset, pid_endpos)}.")

            if len(pids) == 0:
                continue

            print_message(f"###--> Ranking in batches the pairs #{range_start} through #{range_end} in this sub-range.")

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor[tensor_offset:tensor_endpos].to(DEVICE)
            views = self._create_views(collection)

            print_message(f"#> Ranking in batches of {BSIZE} query--passage pairs...")

            for batch_idx, offset in enumerate(range(0, len(pids), BSIZE)):
                if batch_idx % 100 == 0:
                    print_message("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + BSIZE
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]
                Q_weight = all_query_weights[batch_query_index]
                # Q: float tensor (bsize, dim, query_maxlen + exp_embs)
                # Q_weight: float tensor (bsize, query_maxlen + exp_embs)

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset, Q_weight=Q_weight))

        return scores
    
    def get(self, pids):
        """
        Load embeddings for the given pids
        """
        assert len(pids) > 0

        views = self.views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_permutation = [], []
        output_embeddings = []

        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]

            group_offsets = group_offsets.to(VIEWS_DEVICE)
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
            D = D.to(DEVICE)
            D = D[group_offsets_expand.to(DEVICE)].to(dtype=self.maxsim_dtype) # tensor, size (bsize, doc_maxlen, dim)

            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1) # tensor, size (bsize, doc_maxlen)
            D_list = [x[m]for x, m in zip(D, mask)]
            output_embeddings.extend(D_list)

            output_pids.append(group_pids)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices

        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        assert len(raw_pids) == len(output_pids)
        assert raw_pids == output_pids

        output_embeddings = [output_embeddings[_i] for _i in output_permutation.tolist()]

        return output_embeddings
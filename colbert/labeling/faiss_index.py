import os
import faiss
import torch
import numpy as np
from tqdm import tqdm

from colbert.utils.utils import print_message, flatten
from colbert.indexing.loaders import get_parts, load_doclens


def load_tokenids(directory):
    parts, _, _ = get_parts(directory)

    tokenids_filenames = [os.path.join(directory, str(part) + ".tokenids") for part in parts]
    all_tokenids = torch.cat([torch.load(filename) for filename in tokenids_filenames])

    return all_tokenids

def uniq(l):
    return list(set(l))

class FaissIndex():
    def __init__(self, index_path, faiss_index_path, nprobe, part_range=None, inference=None):
        self.inference = inference
        print_message("#> Loading the FAISS index from", faiss_index_path, "..")

        faiss_part_range = os.path.basename(faiss_index_path).split('.')[-2].split('-')

        if len(faiss_part_range) == 2:
            faiss_part_range = range(*map(int, faiss_part_range))
            assert part_range[0] in faiss_part_range, (part_range, faiss_part_range)
            assert part_range[-1] in faiss_part_range, (part_range, faiss_part_range)
        else:
            faiss_part_range = None

        self.part_range = part_range
        self.faiss_part_range = faiss_part_range

        self.faiss_index = faiss.read_index(faiss_index_path)
        self.faiss_index.nprobe = nprobe

        print_message("#> Building the emb2pid mapping..")
        all_doclens = load_doclens(index_path, flatten=False)

        pid_offset = 0
        if faiss_part_range is not None:
            print(f"#> Restricting all_doclens to the range {faiss_part_range}.")
            pid_offset = len(flatten(all_doclens[:faiss_part_range.start]))
            all_doclens = all_doclens[faiss_part_range.start:faiss_part_range.stop]

        self.relative_range = None
        if self.part_range is not None:
            start = self.faiss_part_range.start if self.faiss_part_range is not None else 0
            a = len(flatten(all_doclens[:self.part_range.start - start]))
            b = len(flatten(all_doclens[:self.part_range.stop - start]))
            self.relative_range = range(a, b)
            print(f"self.relative_range = {self.relative_range}")

        all_doclens = flatten(all_doclens)

        total_num_embeddings = sum(all_doclens)
        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid_offset + pid
            offset_doclens += dlength

        print_message("len(self.emb2pid) =", len(self.emb2pid))

        #!@ original
        # self.parallel_pool = Pool(16)

        #!@ custom: Newly added codes for PRF
        print_message("#> Building the emb2tid mapping..")
        self.emb2tid = load_tokenids(index_path)
        self.tok = self.inference.query_tokenizer.tok
        vocab_size = self.tok.vocab_size

        print("Loading doclens")
        part_doclens = load_doclens(index_path, flatten=False)
        self.doclens = np.concatenate([np.array(part) for part in part_doclens])
        self.num_docs = len(self.doclens)
        self.end_offsets = np.cumsum(self.doclens)
        _dfs_file = os.path.join(index_path, 'tokenids.docfreq')
        try:
            self.dfs = torch.load(_dfs_file)
            print(f'\t load document frequencies: {_dfs_file}')
        except:
            dfs = torch.zeros(vocab_size, dtype=torch.int64)
            offset = 0
            for doclen in tqdm(self.doclens, unit="d", desc="Computing document frequencies"):
                tids = torch.unique(self.emb2tid[offset:offset + doclen])
                dfs[tids] += 1
                offset += doclen
            self.dfs = dfs
            torch.save(self.dfs, _dfs_file)
            print(f'\t save document frequencies: {_dfs_file}')
        print("Done")


    def retrieve(self, faiss_depth, Q, verbose=False):
        embedding_ids = self.queries_to_embedding_ids(faiss_depth, Q, verbose=verbose)
        pids = self.embedding_ids_to_pids(embedding_ids, verbose=verbose)

        if self.relative_range is not None:
            pids = [[pid for pid in pids_ if pid in self.relative_range] for pids_ in pids]

        return pids

    def queries_to_embedding_ids(self, faiss_depth, Q, verbose=True):
        # Flatten into a matrix for the faiss search.
        num_queries, embeddings_per_query, dim = Q.size()
        Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()

        # Search in large batches with faiss.
        print_message("#> Search in batches with faiss. \t\t",
                      f"Q.size() = {Q.size()}, Q_faiss.size() = {Q_faiss.size()}",
                      condition=verbose)

        embeddings_ids = []
        faiss_bsize = embeddings_per_query * 5000
        for offset in range(0, Q_faiss.size(0), faiss_bsize):
            endpos = min(offset + faiss_bsize, Q_faiss.size(0))

            print_message("#> Searching from {} to {}...".format(offset, endpos), condition=verbose)

            some_Q_faiss = Q_faiss[offset:endpos].float().numpy()
            _, some_embedding_ids = self.faiss_index.search(some_Q_faiss, faiss_depth) # Search
            embeddings_ids.append(torch.from_numpy(some_embedding_ids))

        embedding_ids = torch.cat(embeddings_ids)

        # Reshape to (number of queries, non-unique embedding IDs per query)
        embedding_ids = embedding_ids.view(num_queries, embeddings_per_query * embedding_ids.size(1))

        return embedding_ids

    def embedding_ids_to_pids(self, embedding_ids, verbose=True):
        # Find unique PIDs per query.
        print_message("#> Lookup the PIDs..", condition=verbose)
        all_pids = self.emb2pid[embedding_ids]

        print_message(f"#> Converting to a list [shape = {all_pids.size()}]..", condition=verbose)
        all_pids = all_pids.tolist()

        print_message("#> Removing duplicates (in parallel if large enough)..", condition=verbose)

        #!@ original
        # if len(all_pids) > 5000:
        #     all_pids = list(self.parallel_pool.map(uniq, all_pids))
        # else:
        #     all_pids = list(map(uniq, all_pids))
        
        #!@ custom
        all_pids = list(map(uniq, all_pids))

        print_message("#> Done with embedding_ids_to_pids().", condition=verbose)

        return all_pids
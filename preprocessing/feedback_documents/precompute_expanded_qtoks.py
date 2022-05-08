import argparse
from collections import OrderedDict
import torch
from tqdm import tqdm

from transformers import BertTokenizerFast

class QueryTokenizer():
    def __init__(self, query_maxlen=32, fb_k=10):
        self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        
        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103

        self.query_maxlen = query_maxlen
        self.fb_k = fb_k
    
    #* Precompute query tokenization: V1: [CLS] [Q] query with [MASK] (32 length) [SEP] expansion terms (10 length)
    def tensorize(self, query, exp):

        batch_text = [query]
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        exp = exp.strip()

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                    return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'][0], obj['attention_mask'][0]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        # for expansion tokens
        if exp:
            batch_text = ['[SEP] ' + exp]
        else:
            batch_text = ['[SEP]']
        
        exp_obj = self.tok(batch_text, padding='max_length', truncation=True, 
                    add_special_tokens=False,
                    return_tensors='pt', max_length=self.fb_k+1) # +1 for [SEP]
        exp_ids, exp_mask = exp_obj['input_ids'][0], exp_obj['attention_mask'][0]
        exp_ids[exp_ids == 0] = self.mask_token_id

        ids = torch.cat([ids, exp_ids,], dim=0)
        mask = torch.cat([mask, exp_mask,], dim=0)
        
        assert len(ids) == self.query_maxlen + 1 + self.fb_k # +1 for [SEP]

        exp_mask = torch.cat([
            torch.zeros(self.query_maxlen+1, dtype=torch.long), # +1 for [SEP]
            torch.ones(self.fb_k, dtype=torch.long),
        ], dim=0)

        toks = self.tok.convert_ids_to_tokens(ids)

        return ids, mask, exp_mask, toks

    

def load_queries(path):
    print(f"#> Loading queries... from: {path}")

    queries = OrderedDict()

    with open(path) as f:
        for line in f:
            qid, query = line.strip().split('\t')
            qid = int(qid)
            queries[qid] = query

    return queries


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--queries', help="path/to/queries.tsv")
    parser.add_argument('--query_maxlen', type=int, default=32)
    parser.add_argument('--fb_k', type=int, default=10)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    tok = QueryTokenizer(query_maxlen=args.query_maxlen, fb_k=args.fb_k)

    queries = load_queries(args.queries)

    qid_to_tensor = OrderedDict()
    for query_index, (qid, query) in enumerate(tqdm(queries.items())):
        
        query, exp = query.strip().split('[SEP]')
        query = query.strip()
        exp = exp.strip()
        
        ids, mask, exp_mask, toks = tok.tensorize(query=query, exp=exp)

        qid_to_tensor[qid] = {'id':ids, 'mask':mask, 'exp_mask':exp_mask, 'toks':toks}

        if query_index < 5:
            print(f'\nqid={qid}, query={query}')
            print(qid_to_tensor[qid])

    torch.save(qid_to_tensor, args.output)
    print(f'\n\n\toutput:\t\t{args.output}\n')


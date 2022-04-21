import argparse
import os
import json

from functools import partial
import string

import torch
from transformers import BertTokenizerFast
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer

def get_parts(directory):
    extension = '.pt'

    parts = sorted([int(filename[: -1 * len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])

    assert list(range(len(parts))) == parts, parts

    # Integer-sortedness matters.
    parts_paths = [os.path.join(directory, '{}{}'.format(filename, extension)) for filename in parts]
    samples_paths = [os.path.join(directory, '{}.sample'.format(filename)) for filename in parts]

    return parts, parts_paths, samples_paths

def get_mask_fn(input_ids, skiplist):
    mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
    return mask

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', default="data/collection.tsv")
    parser.add_argument('--mask-punctuation', dest='mask_punctuation', action='store_true')
    parser.add_argument('--doc_maxlen', type=int, default=180)
    parser.add_argument('--index_path', help="/path/to/index.py/MSMARCO.L2.32x200k/")
    parser.add_argument('--output_path', help="output path", required=True)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    print(f'os.makedirs: {args.output_path}')

    assert args.index_path and os.path.exists(args.index_path)
    parts, _, _ = get_parts(directory=args.index_path)
    doclens_filenames = [os.path.join(args.index_path, f'doclens.{part}.json') for part in parts]
    print(f'#> Load doclens')
    doclens_list = []
    for file in doclens_filenames:
        print(f'#> \tLoad: {file}')
        doclens_list.append(json.load(open(file)))

    print(f'\n#> Init DocTokenizer with doc_maxlen {args.doc_maxlen}')
    tokenizer = DocTokenizer(doc_maxlen=args.doc_maxlen)

    if args.mask_punctuation:
        print(f'\n#> Applying mask_punctuation')
        _tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
        skiplist = {w: True
                for symbol in string.punctuation
                for w in [symbol, _tok.encode(symbol, add_special_tokens=False)[0]]}
        get_mask = partial(get_mask_fn, skiplist=skiplist)
    else:
        get_mask = partial(get_mask_fn, skiplist={})

    
    # Init
    print(f'#> Begin process!')
    ids_2d_list = []
    part_idx = 0
    part = parts[part_idx]
    local_doclens = doclens_list[part_idx]

    with open(args.collection, 'r', encoding='utf-8') as file:
        for line_idx, line in enumerate(file):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)
            
            pid, passage = line.strip().split('\t')
            
            ids, mask = tokenizer.tensorize(batch_text=[passage])

            mask = get_mask(ids)[0]
            # List[bool]
        
            ids = ids[0, :]
            assert len(ids) == len(mask)
            ids = ids[mask]
            # tensor[int]
            
            ids_2d_list.append(ids)

            #?@ debugging
            # if line_idx==10:break 

            if len(ids_2d_list) == len(local_doclens):
                
                for idx, (doclen, tids) in enumerate(zip(local_doclens, ids_2d_list)):
                    assert len(tids) == doclen, (idx, len(tids), doclen)

                # outpath = os.path.join(args.index_path, str(part) + ".tokenids")
                outpath = os.path.join(args.output_path, str(part) + ".tokenids")
                
                torch.save(torch.cat(ids_2d_list), outpath)
                print(f'\n#> Save tokenids: {outpath}')
                
                ids_2d_list = []
                part_idx += 1
                try:
                    part = parts[part_idx]
                    local_doclens = doclens_list[part_idx]
                except:
                    break
    

    print(f'\n#> Done!')






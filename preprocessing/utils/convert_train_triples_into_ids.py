import ujson
import csv
import os
import re
from tqdm import tqdm

from colbert.evaluation.loaders import load_qrels, load_queries, load_collection

printableD =  set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')
printable3D = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')
def cleanQ(s, join=True):
    s = [(x.lower() if x in printable3D else ' ') for x in s]
    s = [(x if x in printableD else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else (' ' if len(min(w.split('-'), key=len)) > 1 else '').join(w.split('-'))) for w in s]
    s = ' '.join(s).split()
    return ' '.join(s) if join else s


printable = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
printableX = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. ')
printable3X = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- ')
def cleanD(s, join=True):
    s = [(x.lower() if x in printable3X else ' ') for x in s]
    s = [(x if x in printableX else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' . ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else w.replace('-', '') + ' ( ' + ' '.join(w.split('-')) + ' ) ') for w in s]
    s = ' '.join(s).split()
    # s = [w for w in s if w not in STOPLIST]
    return ' '.join(s) if join else s

# DATA_DIR = '/workspace/DataCenter/PassageRanking/MSMARCO' # sonic
DATA_DIR = '/workspace/DataCenter/MSMARCO' # dilab003

qrels = load_qrels(DATA_DIR + '/qrels.train.tsv')
queries = load_queries('data/queries.train.reduced.tsv')
print(f'# of qids in qrels = {len(qrels)}')
print(f'# of qids in queries = {len(queries)}')

query2qid = {}
for qid, query in queries.items():
    query = cleanQ(query)
    query2qid[query] = query2qid.get(query, [])
    query2qid[query].append(qid)
print(f'# of unique queries, after cleaning={len(query2qid)}')

collection = load_collection(DATA_DIR+'/collection.tsv')
print(f'# of passages in the document collection = {len(collection)}')
psg2pid = {}
for pid, psg in enumerate(tqdm(collection)):
    psg = cleanD(psg)
    if psg not in psg2pid:
        psg2pid[psg] = pid
print(f'# of unique passages, after cleaning={len(psg2pid)}')





print(f'#> Convert triples of str into triples of ids')
triple_path = DATA_DIR + '/triples.train.small.tsv'
# n_triples = sum(1 for _ in open(triple_path, 'r', encoding='utf-8')) # 39780811
n_triples = 39780811
outfile = open('data/triples.train.small.ids.jsonl', 'w')
with open(triple_path, 'r', encoding='utf-8') as ifile:
    for i_line, line in enumerate(tqdm(ifile, total=n_triples)):
        query, psg1, psg2 = line.strip().split('\t')
        if i_line < 10:
            print((query,psg1,psg2))
        
        query = cleanQ(query)
        assert query in query2qid, f'{query} not in queries'

        rel_qid = -1
        for qid in query2qid[query]:
            if qid in qrels:
                rel_qid = qid
                break
        assert rel_qid!=-1, f'"{query}" (IDs={query2qid[query]}) not in qrels'

        outfile.write(f'{ujson.dumps([rel_qid, psg2pid[cleanD(psg1)], psg2pid[cleanD(psg2)]])}\n')




import ujson
import csv
import os

# DATA_DIR = '/workspace/DataCenter/PassageRanking/MSMARCO' # sonic
DATA_DIR = '/workspace/DataCenter/MSMARCO' # sonic
triple_path = 'data/triples.train.small.ids.jsonl'
qrels_path = DATA_DIR+'/qrels.train.tsv'
quries_path = DATA_DIR + '/queries.train.tsv'
assert os.path.exists(triple_path)
assert os.path.exists(qrels_path)
assert os.path.exists(quries_path)

qid_set2 = set()
with open(qrels_path) as ifile:
    print(f'#> Load {qrels_path}')
    reader = csv.reader(ifile, delimiter='\t')
    for i_row, row in enumerate(reader):
        qid, *others = map(int, row)
        if i_row < 10:
            print(f'qid={qid} ({type(qid)})')
        qid_set2.add(qid)

qid_set3 = set()
with open(quries_path) as ifile:
    print(f'#> Load {quries_path}')
    reader = csv.reader(ifile, delimiter='\t')
    for i_row, row in enumerate(reader):
        qid, *others = row
        qid = int(qid)
        if i_row < 10:
            print(f'qid={qid} ({type(qid)})')
        qid_set3.add(qid)
print(f'# of qids in qrels = {len(qid_set2)}')
print(f'# of qids in queries = {len(qid_set3)}')

qid_set1 = set()
excluded_set2 = set()
excluded_set3 = set()
with open(triple_path) as ifile:
    print(f'#> Load {triple_path}')
    for i_row, line in enumerate(ifile):
        qid, *others = ujson.loads(line)
        # qid = int(qid)
        if i_row < 10:
            print(f'qid={qid} ({type(qid)})')
        qid_set1.add(qid)
        try:
            assert qid in qid_set3, f'{qid} not in queries'
        except:
            print(f'{qid} not in queries')
            excluded_set3.add(qid)
        try:
            assert qid in qid_set2, f'{qid} not in qrels'
        except:
            print(f'{qid} not in qrels')
            excluded_set2.add(qid)
print(f'# of qids in triple = {len(qid_set1)}')
print(f'# of qids in triple & not in qrels = {len(excluded_set2)}')
print(f'# of qids in triple & not in queries = {len(excluded_set3)}')




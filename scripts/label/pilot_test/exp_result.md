# Experimental Setting
- dataset: TREC 2019/2020 evaluation dataset
    - original datasets are
    ```
    data/trec2019:
    2019qrels-docs.txt  2019qrels-pass.txt

    data/trec2020:
    2020qrels-docs.txt  2020qrels-pass.txt
    ```
    - our custom configuration
        - a labeled positive passage: `data/pilot_test/label/[2019/2020]qrels-pass.train.tsv`
        - unlabeled positive passages: `data/pilot_test/label/[2019/2020]qrels-pass.test.tsv`

## Evaluation result (on unlabeled positive passages)

We evaluate retrieval performance using `data/pilot_test/label/[2019/2020]qrels-pass.test.tsv`

### TREC 2019
1. Baselines: using original query
NDCG@10: 0.664, Recall@1k: 0.736, mAP@1k: 0.442

2. Ours: using expanded query with a labeled passage
NDCG@10: 0.681, Recall@1k: 0.751, mAP@1k: 0.474

### TREC 2020
1. Baselines: using original query
NDCG@10: 0.625, Recall@1k: 0.737, mAP@1k: 0.430

2. Ours: using expanded query with a labeled passage
NDCG@10: 0.649, Recall@1k: 0.754, mAP@1k: 0.446

## Evaluation result (on all positive passages)

We evaluate retrieval performance using `data/trec[2019/2020]/[2019/2020]qrels-pass.txt`

### TREC 2019
1. Baselines: using original query
NDCG@10: 0.700, Recall@1k: 0.739, mAP@1k: 0.467

2. Ours: using expanded query with a labeled passage
NDCG@10: 0.754, Recall@1k: 0.754, mAP@1k: 0.516

### TREC 2020
1. Baselines: using original query
NDCG@10: 0.675, Recall@1k: 0.742, mAP@1k: 0.465

2. Ours: using expanded query with a labeled passage
NDCG@10: 0.745, Recall@1k: 0.760, mAP@1k: 0.502



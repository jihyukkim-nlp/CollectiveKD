#!/bin/bash
python -m utility.convert_data.extract_train_triples_from_sebastian_ce_supervision \
--input data/cross_encoder_scores/bertbase_cat_msmarcopassage_train_scores_ids.tsv \
--output data/triples.bm25n.sebastian_single.jsonl

python -m utility.convert_data.extract_train_triples_from_sebastian_ce_supervision \
--input data/cross_encoder_scores/bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv \
--output data/triples.bm25n.sebastian_ensemble.jsonl

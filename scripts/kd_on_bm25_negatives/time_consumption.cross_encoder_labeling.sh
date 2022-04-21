#!/bin/bash
# Execute this on dilab003 (virtualenv: pygaggle)
python -m preprocessing.pseudo_labeling.cross_encoder_labeling_bm25neg \
--reranker MonoBERT --queries data/queries.train.reduced.tsv --triples data/triples.train.small.ids.jsonl \
--collection /workspace/DataCenter/MSMARCO/collection.tsv

#> Loading the queries from data/queries.train.reduced.tsv ...
#> Got 502939 queries. All QIDs are unique.

#> Loading collection...
# 0M 1M 2M 3M 4M 5M 6M 7M 8M 

#> Load MonoBERT
#> [Done] MonoBERT is loaded

#> Load data/triples.train.small.ids.jsonl
#> [Done] data/triples.train.small.ids.jsonl is loaded
#> The # of queries: 502939

#> Sample 100 queries as an approximation

#> Start re-ranking!
# 100%|____________________________________________________________________________________________________________________________________________| 100/100 [02:17<00:00,  1.37s/it]
# Time consumption: 137.07255482673645 seconds for 100
# Approximated time for 502939 queries: 689391.337 seconds = 191.498 hours

# Time consumption: 1331.9278800487518 seconds for 1000
# Approximated time for 502939 queries: 669878.476 seconds = 186.077 hours

# jihyuk@dilab003:~$ nvidia-smi
# Mon Nov  8 03:02:51 2021       
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 440.44       Driver Version: 440.44       CUDA Version: 10.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  GeForce RTX 208...  Off  | 00000000:19:00.0 Off |                  N/A |
# | 57%   58C    P2   249W / 300W |   2450MiB / 11019MiB |     85%      Default |
# +-------------------------------+----------------------+----------------------+
# |   1  GeForce RTX 208...  Off  | 00000000:67:00.0 Off |                  N/A |
# |  0%   45C    P8    23W / 300W |     11MiB / 11019MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
# |   2  GeForce RTX 208...  Off  | 00000000:68:00.0 Off |                  N/A |
# |  0%   39C    P8    18W / 300W |     11MiB / 11016MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
                                                                               
# +-----------------------------------------------------------------------------+
# | Processes:                                                       GPU Memory |
# |  GPU       PID   Type   Process name                             Usage      |
# |=============================================================================|
# |    0     69477      C   python                                      2439MiB |
# +-----------------------------------------------------------------------------+
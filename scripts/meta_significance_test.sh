"""
Significance Test
paired t-test
"""


"""
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 1 --per_query_annotate --queries data/msmarco-pass/queries.dev.small.tsv \
--qrels data/msmarco-pass/qrels.dev.small.tsv --ranking experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/rerank.py/2021-10-12_11.26.35/ranking.tsv
# 
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2019.tsv \
--qrels data/trec2019/2019qrels-pass.txt --ranking experiments/finetuned.b36.lr3e6.hn/TREC2019-psg/rerank.py/2021-10-12_14.24.55/ranking.tsv
# 
python -m utility.evaluate.trec_format_evaluation \
--binarization_point 2 --per_query_annotate --queries data/queries.trec2020.tsv \
--qrels data/trec2020/2020qrels-pass.txt --ranking experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/rerank.py/2021-10-12_14.24.58/ranking.tsv
# 
# 
# 
mkdir -p experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/
scp -P 7777 -r sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/\
experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/rerank.py experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/
# 
mkdir -p experiments/finetuned.b36.lr3e6.hn/TREC2019-psg/
scp -P 7777 -r sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/\
experiments/finetuned.b36.lr3e6.hn/TREC2019-psg/rerank.py experiments/finetuned.b36.lr3e6.hn/TREC2019-psg/
# 
mkdir -p experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/
scp -P 7777 -r sonic@147.46.15.95:/data1/sonic/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/\
experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/rerank.py experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/
"""

#* CFE-HN (1.0) vs ColBERT-HN
# CFE-HN (1.0) vs ColBERT-HN: MSMARCO Dev
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/rerank.py/2022-01-02_21.40.54/ranking.tsv.per_query.metrics \
--perf_b experiments/finetuned.b36.lr3e6.hn/MSMARCO-psg/rerank.py/2021-10-12_11.26.35/ranking.tsv.per_query.metrics
# CFE-HN (1.0) vs ColBERT-HN: TREC-DL 2019
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2019-psg/rerank.py/2022-01-02_21.49.53/ranking.tsv.per_query.metrics \
--perf_b experiments/finetuned.b36.lr3e6.hn/TREC2019-psg/rerank.py/2021-10-12_14.24.55/ranking.tsv.per_query.metrics
# CFE-HN (1.0) vs ColBERT-HN: TREC-DL 2020
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2020-psg/rerank.py/2022-01-02_21.53.48/ranking.tsv.per_query.metrics \
--perf_b experiments/finetuned.b36.lr3e6.hn/TREC2020-psg/rerank.py/2021-10-12_14.24.58/ranking.tsv.per_query.metrics
# 
# 
# 
#* CFE(1.0) vs ColBERT
# CFE(1.0) vs ColBERT: MSMARCO Dev
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/rerank.py/2022-01-02_21.40.54/ranking.tsv.per_query.metrics \
--perf_b experiments/colbert.teacher/MSMARCO-psg/rerank.py/2021-09-11_23.41.17/ranking.tsv.per_query.metrics
# CFE(1.0) vs ColBERT: TREC-DL 2019
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2019-psg/rerank.py/2022-01-02_21.49.53/ranking.tsv.per_query.metrics \
--perf_b experiments/colbert.teacher/TREC2019-psg/2021-11-04_20.35.39/ranking.tsv.per_query.metrics
# CFE(1.0) vs ColBERT: TREC-DL 2020
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2020-psg/rerank.py/2022-01-02_21.53.48/ranking.tsv.per_query.metrics \
--perf_b experiments/colbert.teacher/TREC2020-psg/2021-11-04_20.36.46/ranking.tsv.per_query.metrics
# 
# 
# 
#* CFE-HN (1.0) vs ColBERT
# CFE-HN (1.0) vs ColBERT: MSMARCO Dev
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/rerank.py/2021-10-27_18.16.49/ranking.tsv.per_query.metrics \
--perf_b experiments/colbert.teacher/MSMARCO-psg/rerank.py/2021-09-11_23.41.17/ranking.tsv.per_query.metrics
"""
#* [mrr_10      ] Proposed (0.384) > Baseline (0.367) at p < 0.006
#* [ndcg_10     ] Proposed (0.448) > Baseline (0.430) at p < 0.003
#* [map_1000    ] Proposed (0.389) > Baseline (0.372) at p < 0.005
#* [recall_1000 ] Proposed (0.974) > Baseline (0.967) at p < 0.008
"""
# CFE-HN (1.0) vs ColBERT: TREC-DL 2019
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/TREC2019-psg/rerank.py/2021-10-27_18.27.16/ranking.tsv.per_query.metrics \
--perf_b experiments/colbert.teacher/TREC2019-psg/2021-11-04_20.35.39/ranking.tsv.per_query.metrics
# CFE-HN (1.0) vs ColBERT: TREC-DL 2020
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/TREC2020-psg/rerank.py/2021-10-27_18.32.10/ranking.tsv.per_query.metrics \
--perf_b experiments/colbert.teacher/TREC2020-psg/2021-11-04_20.36.46/ranking.tsv.per_query.metrics
# 
# 
# 
#* CFE (1.0) vs CE-1
# CFE (1.0) vs CE-1: MSMARCO Dev
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/MSMARCO-psg/rerank.py/2022-01-02_21.40.54/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/rerank.py/2022-01-06_09.09.09/ranking.tsv.per_query.metrics
# CFE (1.0) vs CE-1: TREC-DL 2019
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2019-psg/rerank.py/2022-01-02_21.49.53/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2019-psg/rerank.py/2022-01-06_09.21.09/ranking.tsv.per_query.metrics
# CFE (1.0) vs CE-1: TREC-DL 2020
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/kd_on_bm25_negatives/prf.beta1.0.b36.lr3e6.bm25n/TREC2020-psg/rerank.py/2022-01-02_21.53.48/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2020-psg/rerank.py/2022-01-06_09.25.42/ranking.tsv.per_query.metrics
# 
# 
# 
#* CFE-HN (1.0) vs CE-1
# CFE-HN (1.0) vs CE-1: MSMARCO Dev
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/rerank.py/2021-10-27_18.16.49/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_single/MSMARCO-psg/rerank.py/2022-01-06_09.09.09/ranking.tsv.per_query.metrics
"""
#* [mrr_10      ] Proposed (0.384) > Baseline (0.373) at p < 0.051
#* [ndcg_10     ] Proposed (0.448) > Baseline (0.435) at p < 0.021
#* [map_1000    ] Proposed (0.389) > Baseline (0.378) at p < 0.051
#* [recall_1000 ] Proposed (0.974) > Baseline (0.967) at p < 0.007
"""
# CFE-HN (1.0) vs CE-1: TREC-DL 2019
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/TREC2019-psg/rerank.py/2021-10-27_18.27.16/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2019-psg/rerank.py/2022-01-06_09.21.09/ranking.tsv.per_query.metrics
# CFE-HN (1.0) vs CE-1: TREC-DL 2020
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/TREC2020-psg/rerank.py/2021-10-27_18.32.10/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_single/TREC2020-psg/rerank.py/2022-01-06_09.25.42/ranking.tsv.per_query.metrics
"""
[mrr_10      ] Proposed (0.960) > Baseline (0.960) at p < 0.508
[ndcg_10     ] Proposed (0.721) > Baseline (0.696) at p < 0.240
[map_1000    ] Proposed (0.513) > Baseline (0.450) at p < 0.092
[recall_1000 ] Proposed (0.799) > Baseline (0.730) at p < 0.064
"""
# 
# 
# 
#* CFE-HN (1.0) vs CE-3
# CFE-HN (1.0) vs CE-3: MSMARCO Dev
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/MSMARCO-psg/rerank.py/2021-10-27_18.16.49/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/MSMARCO-psg/rerank.py/2022-01-06_07.29.24/ranking.tsv.per_query.metrics
"""
[mrr_10      ] Proposed (0.384) > Baseline (0.378) at p < 0.196
[ndcg_10     ] Proposed (0.448) > Baseline (0.441) at p < 0.141
[map_1000    ] Proposed (0.389) > Baseline (0.384) at p < 0.206
#* [recall_1000 ] Proposed (0.974) > Baseline (0.963) at p < 0.000
"""
# CFE-HN (1.0) vs CE-3: TREC-DL 2019
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/TREC2019-psg/rerank.py/2021-10-27_18.27.16/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2019-psg/rerank.py/2022-01-06_06.29.16/ranking.tsv.per_query.metrics
# CFE-HN (1.0) vs CE-3: TREC-DL 2020
python -m utility.evaluate.trec_format_significance_test \
--perf_a experiments/finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf/TREC2020-psg/rerank.py/2021-10-27_18.32.10/ranking.tsv.per_query.metrics \
--perf_b experiments/kd_on_bm25_negatives/static_kd/ce_ensemble/TREC2020-psg/rerank.py/2022-01-06_06.33.35/ranking.tsv.per_query.metrics
"""
[mrr_10      ] Proposed (0.960) > Baseline (0.937) at p < 0.281
[ndcg_10     ] Proposed (0.721) > Baseline (0.684) at p < 0.174
#* [map_1000    ] Proposed (0.513) > Baseline (0.428) at p < 0.038
#* [recall_1000 ] Proposed (0.799) > Baseline (0.715) at p < 0.040
"""
# 
# 
# 

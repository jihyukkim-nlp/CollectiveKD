#!/bin/bash
device=$1

EXP_ROOT_DIR=experiments/debugging
checkpoint=data/checkpoints/colbert.teacher.dnn

#?@ debugging
#TODO: hn.up_qe.kd_qe
# CUDA_VISIBLE_DEVICES=${device} \
# python -m colbert.train --maxsteps 600000 --amp --bsize 8 --lr 3e-06 --accum 1 \
# --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/triples.train.small.ids.jsonl \
# --queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
# --doc_maxlen 180 --mask-punctuation --similarity l2 \
# --root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
# --knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
# --kd_query_expansion --kd_expansion_pt experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/expansion.pt \
# --checkpoint ${checkpoint}; rm -r experiments/debugging

#?@ debugging
#TODO: hn.up_qe
# CUDA_VISIBLE_DEVICES=${device} \
# python -m colbert.train --maxsteps 600000 --amp --bsize 8 --lr 3e-06 --accum 1 \
# --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs10-exp_beta1.0/label.py/triples.train.small.ids.jsonl \
# --queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
# --doc_maxlen 180 --mask-punctuation --similarity l2 \
# --root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
# --checkpoint ${checkpoint}; rm -r experiments/debugging

#?@ debugging
#TODO: hn.up.kd
# CUDA_VISIBLE_DEVICES=${device} \
# python -m colbert.train --maxsteps 600000 --amp --bsize 8 --lr 3e-06 --accum 1 \
# --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.train.small.ids.jsonl \
# --queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
# --doc_maxlen 180 --mask-punctuation --similarity l2 \
# --root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
# --knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
# --checkpoint ${checkpoint}; rm -r experiments/debugging

#?@ debugging
#TODO: hn.up
# CUDA_VISIBLE_DEVICES=${device} \
# python -m colbert.train --maxsteps 600000 --amp --bsize 8 --lr 3e-06 --accum 1 \
# --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/label.py/triples.train.small.ids.jsonl \
# --queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
# --doc_maxlen 180 --mask-punctuation --similarity l2 \
# --root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
# --checkpoint ${checkpoint}; rm -r experiments/debugging


#?@ debugging
#TODO: hn.kd_qe_kmeans.n4
# CUDA_VISIBLE_DEVICES=${device} \
# python -m colbert.train --maxsteps 600000 --amp --bsize 8 --lr 3e-06 --accum 1 \
# --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn4.jsonl \
# --queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
# --doc_maxlen 180 --mask-punctuation --similarity l2 \
# --root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
# --knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
# --kd_query_expansion --kd_expansion_pt experiments/colbert.teacher/MSMARCO-psg-train-kmeans.k10.beta1.0.clusters10/label.py/2021-10-17_01.17.56/expansion.pt \
# --checkpoint ${checkpoint}; rm -r experiments/debugging


#?@ debugging: finetuned.b18.lr3e6.hn.kd_qe_kmeans.penalty3.n4
# CUDA_VISIBLE_DEVICES=${device} \
# python -m colbert.train --maxsteps 600000 --amp --bsize 8 --lr 3e-06 --accum 1 \
# --triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn4.jsonl \
# --queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
# --doc_maxlen 180 --mask-punctuation --similarity l2 \
# --root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
# --knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
# --kd_query_expansion --kd_expansion_pt experiments/colbert.teacher/MSMARCO-psg-train-kmeans.k10.beta1.0.clusters10/label.py/2021-10-17_01.17.56/expansion.pt \
# --kd_penalty 3.0 \
# --checkpoint ${checkpoint}; rm -r experiments/debugging

#?@ debugging: finetuned.b36.lr3e6.hn.kd_qe_kmeans_prf_beta0.5
CUDA_VISIBLE_DEVICES=${device} \
python -m colbert.train --maxsteps 600000 --amp --bsize 8 --lr 3e-06 --accum 1 \
--triples experiments/colbert.teacher/MSMARCO-psg-train-exp_embs0-exp_beta0/triples.train.small.ids.hn.jsonl \
--queries data/queries.train.reduced.tsv --collection /workspace/DataCenter/PassageRanking/MSMARCO/collection.tsv \
--doc_maxlen 180 --mask-punctuation --similarity l2 \
--root ${EXP_ROOT_DIR} --experiment MSMARCO-psg --run msmarco.psg.l2 \
--knowledge_distillation --kd_temperature 0.25 --teacher_checkpoint ${checkpoint} \
--kd_query_expansion --kd_expansion_pt experiments/colbert.teacher/MSMARCO-psg-train-kmeans.prf_only.docs3.k10.beta0.5.clusters24/label.py/2021-10-25_09.52.14/expansion.pt \
--checkpoint ${checkpoint}; rm -r experiments/debugging
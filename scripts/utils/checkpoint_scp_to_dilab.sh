#!/bin/bash
exp_root=$1 #TODO: input arg
step=$2 #TODO: input arg

checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
scp ${checkpoint} jihyuk@dilab003.yonsei.ac.kr:/hdd/jihyuk/Research/PassageRetrieval/qe_pseudo_labeling/${checkpoint}

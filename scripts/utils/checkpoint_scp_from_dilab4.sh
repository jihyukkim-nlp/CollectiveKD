#!/bin/bash
exp_root=$1 #TODO: input arg
step=$2 #TODO: input arg

checkpoint=${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${step}.dnn
scp -P 7777 sonic@dilab4.snu.ac.kr:/data1/jihyuk/Experiment/PassageRetrieval/qe_pseudo_labeling/${checkpoint} ${checkpoint}

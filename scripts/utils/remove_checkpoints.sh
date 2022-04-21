#!/bin/bash
exp_root=$1 # e.g., "experiments/pqa_colbert-s24-b36-lr3e6"
optimal_step=$2 # e.g., "300000"

# Remove checkpoints, except the optimal checkpoint
echo;echo
echo "Remove checkpoints, except the optimal checkpoint"
echo "  >> ${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${optimal_step}.dnn"
echo;echo
mv -v ${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${optimal_step}.dnn ${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/b-colbert-${optimal_step}.dnn 
rm -v ${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert*
mv -v ${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/b-colbert-${optimal_step}.dnn ${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${optimal_step}.dnn 
ls ${exp_root}/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-${optimal_step}.dnn

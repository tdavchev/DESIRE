#!/bin/bash
export PATH=$PATH:/opt/cuda-7.5.18/bin:/opt/cuDNN-7.0:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s1579267/tools/OpenBLAS/lib:/opt/cuda-7.5.18/lib64:/opt/cuDNN-7.0/lib64:/opt/cuDNN-7.0:
export CUDA_ROOT=/opt/cuda-7.5.18

cd /home/s1045064/deep-learning/DESIRE
source /home/s1045064/dissertation/venv/bin/activate

python /home/s1045064/deep-learning/DESIRE/train.py
#THEANO_FLAGS="device=gpu,mode=FAST_RUN,floatX=float32" python /home/s1045064/CNN_sentence/conv_net_sentence.py -nonstatic -word2vec

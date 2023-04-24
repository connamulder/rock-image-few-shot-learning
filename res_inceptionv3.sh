#!/bin/bash
# ( 'baseline' 'cosine' 'pearson' )
# When we specify "--use_knowledge_propagation", we don't need to assign value, the default value is True;
# if "--use_knowledge_propagation" is not specified, the default value is False.

arr=( 'baseline' 'cosine' 'pearson' )
maxiters=1001
OutDir='F:/Pytorch_learning/rock-image-few-shot-learning/results/InceptionV3'
TrainFile='F:/Pytorch_learning/rock-image-few-shot-learning/data/InceptionV3_train.hdf5'
ValFile='F:/Pytorch_learning/rock-image-few-shot-learning/data/InceptionV3_valid.hdf5'
TestFile='F:/Pytorch_learning/rock-image-few-shot-learning/data/InceptionV3_test.hdf5'


# Low-shot benchmark without generation
for j in "${arr[@]}"
do
  # for i in 1 2 5 10 20 21
  for i in {1..2..1}
  do

    echo $j
    if [ $j == "baseline" ];then
    CUDA_VISIBLE_DEVICES=$1 python main_learning.py \
      --lowshotn $i \
      --classifier_type ${j} \
      --outdir $OutDir \
      --trainfile $TrainFile \
      --valfile $ValFile \
      --testfile $TestFile \
      --maxiters $maxiters
    else
    CUDA_VISIBLE_DEVICES=$1 python main_learning.py \
      --lowshotn $i \
      --classifier_type ${j} \
      --outdir $OutDir \
      --trainfile $TrainFile \
      --valfile $ValFile \
      --testfile $TestFile \
      --use_knowledge_propagation \
      --maxiters $maxiters
    fi
  done
done
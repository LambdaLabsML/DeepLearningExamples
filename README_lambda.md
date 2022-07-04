# PyTorch

## Classification

### resnet50

```
cd /home/ubuntu/repos/DeepLearningExamples/PyTorch/Classification/ConvNets

docker build . -t nvidia_resnet50


docker run --gpus 1 \
--rm -it \
-v /home/ubuntu/repos/DeepLearningExamples/PyTorch/Classification/ConvNets:/code \
-v /home/ubuntu/repos/DeepLearningExamples/lambdalabs:/lambdalabs \
--ipc=host \
--workdir=/code \
nvidia_resnet50

export BATCH_SIZE=128
export LAMBDA_LOG_BATCH_SIZE=$BATCH_SIZE
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/resnet50/QuadroRTX8000/FP32 

python ./launch.py --model resnet50 \
--precision FP32 \
--mode benchmark_training \
--platform DGX1V \
/data/imagenet \
--raport-file benchmark.json \
--epochs 1 --prof 20 \
--batch-size $BATCH_SIZE \
--data-backend synthetic && \
chmod -R 777 $LAMDBA_LOG_DIR

```

### efficientnet

Need to check the `config.yml` file and see what `platform` is available for the `model`. For example, `DGX1V` is not available for `efficientnet-b0`.

```
export BATCH_SIZE=128
export LAMBDA_LOG_BATCH_SIZE=$BATCH_SIZE
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/efficientnet-b0/QuadroRTX8000/FP32 

python ./launch.py --model efficientnet-b0 \
--precision FP32 \
--mode benchmark_training \
--platform DGX1V-16G \
/data/imagenet \
--raport-file benchmark.json \
--epochs 1 --prof 20 \
--batch-size $BATCH_SIZE \
--data-backend synthetic && \
chmod -R 777 $LAMDBA_LOG_DIR
```


## Detection

### SSD

```
cd /home/ubuntu/repos/DeepLearningExamples/PyTorch/Detection/SSD
./download_dataset.sh ~/data-deeplearningexamples

docker build . -t nvidia_ssd

docker run --gpus 1 \
--rm -it \
-v /home/ubuntu/repos/DeepLearningExamples/PyTorch/Detection/SSD:/code \
-v /home/ubuntu/repos/DeepLearningExamples/lambdalabs:/lambdalabs \
-v /home/ubuntu/data-deep-learning-examples/mscoco:/coco \
--ipc=host \
--workdir=/code \
nvidia_ssd

export BATCH_SIZE=128
export LAMBDA_LOG_BATCH_SIZE=$BATCH_SIZE
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/ssd/QuadroRTX8000/FP32

python -m torch.distributed.launch --nproc_per_node=1 \
       main.py --batch-size $BATCH_SIZE \
               --mode benchmark-training \
               --benchmark-warmup 20 \
               --benchmark-iterations 40 \
               --data /coco
```


## DrugDiscovery

### SE3Transformer

TODO: need to mount the deeplearning repo and make sure logs are written correctly

mkdir -p ~/temp/results

```
cd /home/ubuntu/repos/DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer

docker build -t se3-transformer .

docker run --gpus 1 \
--rm -it \
--shm-size=8g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /home/ubuntu/repos/DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer:/code \
-v /home/ubuntu/repos/DeepLearningExamples/lambdalabs:/lambdalabs \
--workdir=/code \
--rm se3-transformer:latest


export BATCH_SIZE=32
export AMP=true
export NUM_GPU=1 
export LAMBDA_LOG_BATCH_SIZE=$BATCH_SIZE
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/se3-transformer/QuadroRTX8000/FP32

python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPU --max_restarts 0 --module \
  se3_transformer.runtime.training \
  --amp $AMP \
  --batch_size $BATCH_SIZE \
  --epochs 1 \
  --use_layer_norm \
  --norm \
  --save_ckpt_path model_qm9.pth \
  --task homo \
  --precompute_bases \
  --seed 42 \
  --benchmark



CUDA_VISIBLE_DEVICES=0 python -m se3_transformer.runtime.training \
  --amp $AMP \
  --batch_size $BATCH_SIZE \
  --epochs 1 \
  --use_layer_norm \
  --norm \
  --save_ckpt_path model_qm9.pth \
  --task homo \
  --precompute_bases \
  --seed 42 \
  --benchmark
```

## Forecasting

### TFT

```
docker run --gpus 1 \
--rm -it \
--shm-size=8g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /home/ubuntu/repos/DeepLearningExamples/PyTorch/Forecasting/TFT:/code \
-v /home/ubuntu/repos/DeepLearningExamples/lambdalabs:/lambdalabs \
-v /home/ubuntu/data-deep-learning-benchmark/pytorch/forcasting/tft:/data/ \
--workdir=/code \
tft


export SEED=1
export LR=1e-3
export NGPU=1
export BATCH_SIZE=1024
export EPOCHS=1
export LAMBDA_LOG_BATCH_SIZE=$BATCH_SIZE
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/TFT_electricity_bin/QuadroRTX8000/FP32

python -m torch.distributed.run --nproc_per_node=${NGPU} train.py \
        --dataset electricity \
        --data_path /data/processed/electricity_bin \
        --batch_size=${BATCH_SIZE} \
        --sample 450000 50000 \
        --lr ${LR} \
        --epochs 1 \
        --seed ${SEED} \
        --use_amp \
        --results /results/TFT_electricity_bs${NGPU}x${BATCH_SIZE}_lr${LR}/seed_${SEED}

export SEED=1
export LR=1e-3
export NGPU=1
export BATCH_SIZE=1024
export EPOCHS=1
export LAMBDA_LOG_BATCH_SIZE=$BATCH_SIZE
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/TFT_traffic_bin/QuadroRTX8000/FP32

python -m torch.distributed.run --nproc_per_node=${NGPU} train.py \
        --dataset traffic \
        --data_path /data/processed/traffic_bin \
        --batch_size=${BATCH_SIZE} \
        --sample 450000 50000 \
        --lr ${LR} \
        --epochs ${EPOCHS} \
        --seed ${SEED} \
        --use_amp \
        --results /results/TFT_traffic_bs${NGPU}x${BATCH_SIZE}_lr${LR}/seed_${SEED}
```


## Language Modeling

### BERT

```
export DATA_DIR=~/data-deeplearningexamples
mkdir -p ${DATA_DIR}/data/squad

cd /home/ubuntu/repos/DeepLearningExamples/PyTorch/LanguageModeling/BERT
pushd .
./data/squad/squad_download.sh ${DATA_DIR}/data/squad
popd
chmod -R a+rwx ${DATA_DIR}/data/squad


Pre-trained BERT model from this link
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide


For SQuAd should be the QA models:
https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_qa_squad11_amp/files
https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_qa_squad11_amp/files


pushd .
mkdir -p ${DATA_DIR}/data/bert_large
cd ${DATA_DIR}/data/bert_large
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_large/bert_large_qa.pt
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_large/bert_config.json
wget https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_large/bert-large-uncased-vocab.txt
popd
chmod -R a+rwx ${DATA_DIR}/data/bert_large

pushd .
mkdir -p ${DATA_DIR}/data/bert_base
cd ${DATA_DIR}/data/bert_base
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_base/bert_base_qa.pt
curl -LO https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_base/bert_config.json
wget https://lambdalabs-files.s3-us-west-2.amazonaws.com/bert_base/bert-base-uncased-vocab.txt
popd
chmod -R a+rwx ${DATA_DIR}/data/bert_base


# Build the docker image
bash scripts/docker/build.sh


# Launch the container 
export DOCKER_BRIDGE=host
export IMAGE=bert
export DATA_DIR=${HOME}/data-deeplearningexamples/data
export RESULTS_DIR=${HOME}/data-deeplearningexamples/results

mkdir -p $RESULTS_DIR

docker run -it --rm \
--gpus 1 \
--net=$DOCKER_BRIDGE \
--shm-size=8g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v $DATA_DIR:/data \
-v $RESULTS_DIR:/results \
-v /home/ubuntu/repos/DeepLearningExamples/PyTorch/LanguageModeling/BERT:/code \
-v /home/ubuntu/repos/DeepLearningExamples/lambdalabs:/lambdalabs \
--workdir=/code \
$IMAGE

# Run the jobs 

export NUM_GPU=1 
export LAMBDA_LOG_BATCH_SIZE=4

# BERT Base
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/LanguageModeling_bert_base_SQuAD/QuadroRTX8000/FP32
export init_checkpoint=/data/bert_base/bert_base_qa.pt
export epochs=1
export batch_size=${LAMBDA_LOG_BATCH_SIZE}
export learning_rate=3e-5
export warmup_proportion=0.1
export precision=fp32
export NUM_GPU=1
export seed=1
export squad_dir=/data/squad/v1.1
export vocab_file=/data/bert_base/bert-base-uncased-vocab.txt
export OUT_DIR=/results/SQuAD
export mode=train
export CONFIG_FILE=/data/bert_base/bert_config.json
export max_steps=100


# BERT Large
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/LanguageModeling_bert_large_SQuAD/QuadroRTX8000/FP32
export init_checkpoint=/data/bert_large/bert_large_qa.pt
export epochs=1
export batch_size=${LAMBDA_LOG_BATCH_SIZE}
export learning_rate=3e-5
export warmup_proportion=0.1
export precision=fp32
export NUM_GPU=1
export seed=1
export squad_dir=/data/squad/v1.1
export vocab_file=/data/bert_large/bert-large-uncased-vocab.txt
export OUT_DIR=/results/SQuAD
export mode=train
export CONFIG_FILE=/data/bert_large/bert_config.json
export max_steps=100


bash scripts/run_squad.sh \
$init_checkpoint \
$epochs \
$batch_size \
$learning_rate \
$warmup_proportion \
$precision \
$NUM_GPU \
$seed \
$squad_dir \
$vocab_file \
$OUT_DIR \
$mode \
$CONFIG_FILE \
$max_steps

```

### BART

```
# Build image
bash scripts/docker/build.sh


# Launch docker
export DATA_DIR=~/data-deeplearningexamples

mkdir -p ${DATA_DIR}/data/bart

docker run -it --rm \
--gpus 1 \
--ipc=host \
--shm-size=8g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v ${DATA_DIR}/data/bart:/data \
-v /home/ubuntu/repos/DeepLearningExamples/lambdalabs:/lambdalabs \
-v ${PWD}:/code \
--workdir=/code bart_pyt

# Download data
bash scripts/get_data.sh /data

# Run job
export LAMBDA_LOG_BATCH_SIZE=12
export MAX_SOURCE_LEN=1024
export MAX_TARGET_LEN=60
export DATA_DIR=/data/xsum
export NUM_GPU=1
export precision=fp32
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/LanguageModeling_bart_xsum/QuadroRTX8000/FP32


export LAMBDA_LOG_BATCH_SIZE=24
export MAX_SOURCE_LEN=1024
export MAX_TARGET_LEN=60
export DATA_DIR=/data/xsum
export NUM_GPU=1
export precision=fp16
export LAMDBA_LOG_DIR=/lambdalabs/PyTorch/LanguageModeling_bart_xsum/QuadroRTX8000/FP16


bash scripts/run_training_benchmark.sh \
$LAMBDA_LOG_BATCH_SIZE \
$MAX_SOURCE_LEN \
$MAX_TARGET_LEN \
$DATA_DIR \
$NUM_GPU \
$precision



```


# Caveats

## Have a mixed GPU servers can be a problem

Some libraries, such as `dgl`, does auto detection of GPU architecture and build for a specific version. The build works for the first GPU it detected, and may not work for the others if they are a different sm generation.
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


# Caveats

## Have a mixed GPU servers can be a problem

Some libraries, such as `dgl`, does auto detection of GPU architecture and build for a specific version. The build works for the first GPU it detected, and may not work for the others if they are a different sm generation.
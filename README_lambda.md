# PyTorch

## resnet50

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
--data-backend synthetic

chmod -R 777 $LAMDBA_LOG_DIR
```

## efficientnet

Need to check the `config.yml` file and see what `platform` is available for the `model`. For example, `DGX1V` is not available for `efficientnet-b0`.

```
python ./launch.py --model efficientnet-b0 \
--precision FP32 \
--mode benchmark_training \
--platform DGX1V-16G \
/data/imagenet \
--raport-file benchmark.json \
--epochs 1 --prof 20 \
--batch-size $BATCH_SIZE \
--data-backend synthetic
```
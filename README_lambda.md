# PyTorch

## resnet50

```
docker run --gpus 1 \
--rm -it \
-v /home/ubuntu/repos/DeepLearningExamples/PyTorch/Classification/ConvNets:/code \
-v /home/ubuntu/repos/DeepLearningExamples/lambdalabs:/lambdalabs \
--ipc=host \
--workdir=/code \
nvidia_resnet50

BATCH_SIZE=128 && \
LAMBDA_LOG_BATCH_SIZE=$BATCH_SIZE && \
LAMDBA_LOG_DIR=/lambdalabs/PyTorch/resnet50/QuadroRTX8000/FP32 

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


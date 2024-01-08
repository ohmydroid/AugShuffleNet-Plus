# AugShuffleNet-plus

This repo presents a faster version of AugShuffleNet. 

Model Acceleration comes from two aspects:

1. We remove channel shuffle operation widely used in ShuffleNetV2 and AugShuffleNet, offering obvious improvement over training/inference speed. By enabling FIFO mode of feature rerangement, AugSHuffleNet-Plus allows parallel computing for both training and inference process.

2. By replacing $K\times K$ depth-wise separable convolution with $K\times K$ regular convolution, it can be also a potential GPU-friendly model, which does not rely on residual connection and channel shuffle.


```python
python main.py --model augshufflenet
```


# Results
## 10 runs on cifar10
|Model |               MAdds | Params  | Acc(%)          |                                              training time/epoch| 
|----------------------|------|---------|-----|-----|
|AugShuffleNet 1.5x    |85.38M |2.22M     |94.00  93.95 94.16 93.98 93.86 94.07 93.94 93.83 93.45 94.12  |21.4s|
|ShuffleNetV2  1.5x    |94.27M |2.49M     |93.92  93.47 93.28 93.59 93.67 93.71 93.78 93.55 93.44 93.72  |26.2s|
|AugShuffleNet 1.0x    |43.55M |1.21M    |93.32  93.51 93.38 93.50 93.39 93.50 93.43 93.13 93.27 93.44   |14.7s|
|ShuffleNetV2  1.0x    |45.01M |1.26M    |92.88  92.98 92.75 92.93 92.87 92.87 93.14 92.69 92.80 93.05   |17.5s|
|AugShuffleNet 0.5x    |10.20M |0.33M   |90.24  90.75 91.17 90.46 91.01 91.29 90.92 91.03 91.20 90.91   |7.3s|
|ShuffleNetV2  0.5x    |10.91M |0.35M    |90.23  90.74 90.56 90.58 90.31 89.96 90.66 89.78 90.76 90.05   |8.8s|




## 10 runs on cifar100
|Model |               MAdds | Params  | Acc           |                                              training time/epoch| 
|----------------------|------|---------|-----|-----|
|AugShuffleNet 1.5x|  85.5M  |2.32M   |74.12   74.55 74.95 74.88 74.78 74.27 74.90 74.61 74.36 74.68 |21.4s|
|ShuffleNetV2 1.5x|   94.36M |2.58M  |73.47 73.68  73.64 73.76 74.10 73.74 73.58 73.87  73.35 73.35|26.2s|
|AugShuffleNet  1.0x|  43.65M |1.30M    |73.65   72.87 72.82 72.81 73.38 73.54 72.57 73.19 72.41 73.09 |14.7s|
|ShuffleNetV2 1.0x  | 45.10M |1.36M  |71.70 71.97  72.11 71.91 71.62 71.95 72.22 71.49  72.08 72.28 |17.5s|
|AugShuffleNet 0.5x  |10.29M |0.42M   | 68.95   68.67 68.90 68.28 68.48 68.20 68.22 69.45 68.04 68.06 |7.3s|
|ShuffleNetV2 0.5x   |11.00M |0.44M  |67.39 66.70  67.47 66.12 66.55 67.44 66.74 66.82  66.52 66.76 |8.8s|


Above results presents the training speed of AugSHuffleNet-plus without channel shuffle. Parallel computing supported by FIFO mode will reduce more latency in the future.

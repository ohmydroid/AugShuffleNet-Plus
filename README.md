# AugShuffleNet-plus
Faster version of AugShuffleNet, has better performance than shufflenetv2 with less computational cost and  higher training/inference speed.

There is no channel shuffle, AugShuffleNet-Plus computes partially, crossover swiftly. By replacing  depth-wise convolution with regular convolution, it can be also a potential GPU-friendly model, which does not rely on residual connection and channel shuffle.

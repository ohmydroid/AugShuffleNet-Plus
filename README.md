# AugShuffleNet-plus

Faster version of AugShuffleNet, has better performance than shufflenetv2 with less computational cost and  higher training/inference speed.

There is no channel shuffle, AugShuffleNet-Plus computes partially, crossovers swiftly, allowing us to construct deep networks in an efficient way.

By replacing $K\times K$ depth-wise separable convolution with $K\times K$ regular convolution, it can be also a potential GPU-friendly model, which does not rely on residual connection and channel shuffle.


```python
python main.py --model augshufflenet
```

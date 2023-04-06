## LightGCN-pytorch

This is the Pytorch implementation for SIGIR 2020 paper:

> SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Author: Prof. Xiangnan He (staff.ustc.edu.cn/~hexn/)

I just made a little modification for myself on the basis of https://github.com/gusye1234/LightGCN-PyTorch, for my own reading and reuse.


(Also see Tensorflow [implementation](https://github.com/kuandeng/LightGCN))

(Also see PyTorch [implementation](https://github.com/gusye1234/LightGCN-PyTorch))

The output of my model, I checked, is exactly the same as them.

## An example to run a 3-layer LightGCN

run LightGCN on **Gowalla** dataset:

```python
python main.py --decay=1e-4 --lr=0.001 --layers=3 --seed=2020 --dataset="gowalla" --topks="[20]" --embedding_size=64
```


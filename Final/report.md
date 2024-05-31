# Report

## Torch 配置

由于电脑上没有 [合适的 GPU](https://developer.nvidia.com/cuda-gpus#compute)，故采用 CPU 推理。参考 PyTorch 的 [官方教程](https://pytorch.org/get-started/locally/) 进行环境配置。

### 虚拟环境

```sh
py -m venv --symlinks --upgrade-deps .venv
```

### 安装库

```sh
pip install numpy torch torchvision torchaudio
```

![](assets/Pasted%20image%2020240531143758.png)

![](assets/Pasted%20image%2020240531143815.png)

### 验证安装

![](assets/Pasted%20image%2020240531144131.png)

## 实验原理

> 二元零和马尔可夫博弈，Naive Self Play，Actor-Critic 等；请用自己的话进行阐述，不要照抄 `readme.md`

## 模型设计

> 附上对应 `__init__` 和 `forward` 方法中的关键代码，解释这样设计模型的动机和理由；

## 解决思路

> 着重分析 constrained policy 的解决思路，即 `forward` 的部分是如何解决置 0 限制和归一限制的

## Bug 解决

> `optimize` 函数中，2 个 bug 的解决方案；在设计模型和处理输入输出时，自己遇到的其他 bug（比如因为粗心导致张量形状匹配不上，或者 in-place 操作导致梯度无法计算等等 **自己所有可能遇到的问题**）的解决方案；

## 实验过程中的疑惑

> 记录实验中所有最初自己看似奇怪或难以理解的点。完成实验后，尝试自行解答这些问题，或给出潜在的改进方案。

## 分析并解释 loss 和 entropy 曲线

## 选做 1

> 结合 loss 的变化和模型的评测表现，进行超参数的选择和模型纵深的调整；或者尝试不同的模型架构（cnn，注意力机制等）

## 选做 2

> 思考题：在最理想的情况下，假设我们的模型的权重能够收敛到使得 actor loss 和 critic loss 最优的点。此时黑棋和白棋双方的策略是否一定达成纳什均衡（Nash equilibrium）？为什么？

## 课程反馈

## 参考



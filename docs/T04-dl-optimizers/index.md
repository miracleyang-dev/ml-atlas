# 【T04】DL Optimizers 深度学习优化器

---

## 00 前置记号与基础结论

### 记号约定

| 符号 | 含义 |
| --- | --- |
| $\theta_t$ | 第 $t$ 步模型参数 |
| $L(\theta)$ | 全量训练目标或经验风险 |
| $g_t$ | 第 $t$ 步随机梯度估计 |
| $\eta_t$ | 学习率 |
| $m_t$ | 一阶动量 / 梯度均值估计 |
| $v_t$ | 二阶矩 / 梯度平方均值估计 |
| $\beta,\beta_1,\beta_2$ | 指数滑动平均衰减系数 |
| $\epsilon$ | 防止除零的小常数 |
| $\lambda$ | 权重衰减系数 |
| $B$ | batch size |
| $\|g\|$ | 梯度范数 |

深度学习优化器的目标是近似求解：

$$
\min_\theta \; L(\theta)=\mathbb{E}_{(x,y)\sim \mathcal{D}}[\ell(f_\theta(x),y)].
$$

实际训练使用 mini-batch 梯度：

$$
g_t=\frac{1}{B}\sum_{i\in\mathcal{B}_t}\nabla_\theta \ell(f_{\theta_t}(x_i),y_i),
\qquad
\mathbb{E}[g_t]\approx \nabla L(\theta_t).
$$

### 必备前置结论

**一阶下降近似**：

$$
L(\theta+\Delta)\approx L(\theta)+\nabla L(\theta)^\top \Delta.
$$

取 $\Delta=-\eta \nabla L(\theta)$ 可使一阶项下降。

**光滑函数下降引理**：若 $L$ 的梯度是 $\beta$-Lipschitz，则

$$
L(\theta-\eta\nabla L(\theta))
\le L(\theta)-\eta\left(1-\frac{\beta\eta}{2}\right)\|\nabla L(\theta)\|^2.
$$

当 $0<\eta<2/\beta$ 时，梯度下降具有单步下降保证。

**随机梯度噪声**：mini-batch 梯度可写为

$$
g_t=\nabla L(\theta_t)+\xi_t,
\qquad
\mathbb{E}[\xi_t]=0.
$$

优化器的很多设计都在处理这个噪声：平滑、缩放、裁剪或调度学习率。

---

## 01 问题定义与模型设定

深度学习优化和经典凸优化不同：

1. 目标通常非凸；
2. 参数维度巨大；
3. 梯度来自 mini-batch，有随机噪声；
4. Hessian 无法显式存储；
5. 泛化性能不只取决于训练 loss；
6. 硬件吞吐和数值精度会影响算法选择。

### 1.1 统一更新框架

多数一阶优化器都可写成

$$
\theta_{t+1}=\theta_t+\Delta_t,
$$

其中 $\Delta_t$ 由当前梯度、历史梯度、学习率调度、正则化和数值缩放共同决定。

### 1.2 学习率是主控制旋钮

学习率过大：

- loss 震荡；
- 参数发散；
- mixed precision 下出现 NaN；
- Adam 的二阶矩估计来不及稳定。

学习率过小：

- 训练进展慢；
- 卡在平台区；
- warmup 后有效更新不足。

### 1.3 为什么不用纯 Newton 法

Newton 更新为

$$
\theta_{t+1}=\theta_t-H_t^{-1}g_t.
$$

大模型中 $H_t$ 的维度是参数量平方，无法显式存储或分解。实际使用的一阶优化器可以看作对曲率、动量和噪声的廉价近似。

---

## 02 完整数学推导

### 2.1 Vanilla SGD

最基础的随机梯度下降：

$$
\boxed{\theta_{t+1}=\theta_t-\eta_t g_t.}
$$

优点：

- 实现简单；
- 内存开销低；
- 噪声有时有助于逃离尖锐极小值；
- 在视觉任务和大 batch 训练中仍很常用。

缺点：

- 对学习率敏感；
- 在病态曲率下震荡；
- 缺乏方向平滑。

### 2.2 Momentum

Momentum 引入速度变量：

$$
v_t=\mu v_{t-1}+g_t,
$$

$$
\boxed{\theta_{t+1}=\theta_t-\eta_t v_t.}
$$

展开可得

$$
v_t=g_t+\mu g_{t-1}+\mu^2g_{t-2}+\cdots.
$$

因此 momentum 是历史梯度的指数加权和。它能在一致方向上加速，在震荡方向上相互抵消。

常见写法也会把学习率并入速度：

$$
v_t=\mu v_{t-1}-\eta_t g_t,
\qquad
\theta_{t+1}=\theta_t+v_t.
$$

二者只差变量定义。

### 2.3 Nesterov Momentum

Nesterov 的思想是先看一步未来位置，再计算梯度：

$$
v_t=\mu v_{t-1}+\nabla L(\theta_t-\eta\mu v_{t-1}),
$$

$$
\theta_{t+1}=\theta_t-\eta v_t.
$$

直观上，它在高速前进前先检查前方坡度，因此在凸优化中有更好的理论性质。深度学习框架中常通过等价变形实现。

### 2.4 AdaGrad

AdaGrad 累积历史平方梯度：

$$
s_t=s_{t-1}+g_t\odot g_t.
$$

更新为

$$
\boxed{\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}.}
$$

每个参数都有自己的有效学习率。经常出现大梯度的参数会被更强地缩放。

问题是 $s_t$ 单调增加，学习率会不断变小，深度网络训练后期可能过早停滞。

### 2.5 RMSProp

RMSProp 把 AdaGrad 的累积和改成指数滑动平均：

$$
v_t=\rho v_{t-1}+(1-\rho)g_t^2.
$$

更新：

$$
\boxed{\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}.}
$$

它保留自适应缩放，又避免学习率无限衰减。

### 2.6 Adam

Adam 同时估计一阶矩和二阶矩：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)(g_t\odot g_t).
$$

由于 $m_0=v_0=0$，早期估计偏向零，需要偏差修正：

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}.
$$

更新：

$$
\boxed{\theta_{t+1}=\theta_t-\eta_t\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.}
$$

Adam 的默认超参常为

$$
\beta_1=0.9,
\qquad
\beta_2=0.999,
\qquad
\epsilon=10^{-8}.
$$

### 2.7 AdamW：解耦权重衰减

传统 L2 正则把目标改成

$$
L_{reg}(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2,
$$

梯度变为

$$
g_t^{reg}=g_t+\lambda\theta_t.
$$

在 SGD 中，L2 正则等价于权重衰减。但在 Adam 中，由于梯度会被 $\sqrt{v_t}$ 自适应缩放，二者不再等价。

AdamW 使用解耦权重衰减：

$$
\theta_{t+1}=\theta_t-\eta_t\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}-\eta_t\lambda\theta_t.
$$

也可写成

$$
\boxed{\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.}
$$

这是 Transformer / LLM 训练中最常见的默认选择。

### 2.8 AMSGrad

Adam 在某些凸问题上可能不收敛。AMSGrad 修正二阶矩：

$$
\tilde{v}_t=\max(\tilde{v}_{t-1},\hat{v}_t),
$$

$$
\theta_{t+1}=\theta_t-\eta_t\frac{\hat{m}_t}{\sqrt{\tilde{v}_t}+\epsilon}.
$$

它保证分母不减小，避免有效学习率异常增大。

### 2.9 Lion

Lion 使用动量的符号方向更新：

$$
u_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
$$

$$
\boxed{\theta_{t+1}=\theta_t-\eta_t\operatorname{sign}(u_t).}
$$

随后更新动量缓存：

$$
m_t=\beta_2 m_{t-1}+(1-\beta_2)g_t.
$$

Lion 的状态量比 Adam 少，不维护二阶矩；但学习率和权重衰减通常需要重新调。

---

## 03 几何直观解释

### SGD：带噪声的下坡

SGD 每一步只看 mini-batch 梯度，因此方向有噪声。噪声会让路径抖动，但也可能帮助离开尖锐局部结构。

### Momentum：低通滤波器

Momentum 对历史梯度做指数平滑。若某方向梯度长期一致，速度累积；若方向来回震荡，正负抵消。这解释了它在狭长峡谷中比 SGD 更稳定。

### 自适应优化器：坐标级预条件

Adam / RMSProp 把每个坐标除以历史梯度平方根：

$$
\Delta_i \propto \frac{g_i}{\sqrt{v_i}+\epsilon}.
$$

梯度尺度大的坐标被压缩，尺度小的坐标被相对放大。这类似对角预条件矩阵。

### Weight decay：持续拉回参数

解耦权重衰减每步把参数乘以 $1-\eta\lambda$。它不直接依赖当前梯度，而是给模型容量一个持续约束。

---

## 04 核心性质与关键定理

### 4.1 SGD 的收敛直觉

在光滑凸函数、无偏梯度、方差有界等条件下，SGD 使用递减学习率可收敛到最优点附近。非凸情形常讨论梯度范数：

$$
\mathbb{E}\|\nabla L(\theta_R)\|^2
$$

随训练步数下降，表示趋近一阶驻点。

### 4.2 Momentum 的有效窗口

指数滑动平均

$$
m_t=\beta m_{t-1}+(1-\beta)g_t
$$

大致对应长度为

$$
\frac{1}{1-\beta}
$$

的历史窗口。$\beta=0.9$ 约等于看最近 10 步，$\beta=0.99$ 约等于看最近 100 步。

### 4.3 Adam 偏差修正的必要性

若 $m_0=0$，且梯度期望恒定为 $\mu_g$，则

$$
\mathbb{E}[m_t]=(1-\beta_1^t)\mu_g.
$$

不除以 $1-\beta_1^t$，早期一阶矩会系统性偏小。二阶矩同理。

### 4.4 AdamW 与 L2 正则的区别

SGD 中：

$$
\theta_{t+1}=\theta_t-\eta(g_t+
\lambda\theta_t)=(1-\eta\lambda)\theta_t-\eta g_t.
$$

因此 L2 正则等价于权重衰减。

Adam 中：

$$
\frac{g_t+\lambda\theta_t}{\sqrt{v_t}+\epsilon}
$$

会让正则项也被自适应分母缩放，导致每个坐标衰减强度不同。AdamW 则直接对参数衰减。

### 4.5 Batch size 与梯度噪声

batch size 增大时，梯度方差通常下降：

$$
\mathrm{Var}(g_B)\approx \frac{1}{B}\mathrm{Var}(g_1).
$$

大 batch 训练常配合学习率线性缩放：

$$
\eta_B \approx \eta_{B_0}\frac{B}{B_0},
$$

但超过临界 batch size 后收益会下降，需要 warmup 和更谨慎的调度。

### 4.6 学习率调度

常见调度：

| 调度 | 形式 | 典型场景 |
| --- | --- | --- |
| Step decay | 每隔若干 epoch 乘常数 | CNN 传统训练 |
| Exponential decay | $\eta_t=\eta_0\gamma^t$ | 简单稳定任务 |
| Cosine decay | 余弦下降到最小值 | Transformer、视觉模型 |
| Warmup + cosine | 先升后降 | 大 batch / AdamW |
| One-cycle | 先升后降并调 momentum | 快速训练探索 |

Warmup 的作用是让动量和二阶矩估计先稳定，再使用峰值学习率。

---

## 05 工程实战要点

### 5.1 默认选择

| 模型 / 任务 | 常见优化器 | 备注 |
| --- | --- | --- |
| Transformer / LLM | AdamW | 配合 warmup + cosine |
| CNN 图像分类 | SGD + Momentum 或 AdamW | 大规模训练常用 SGD |
| 小数据微调 | AdamW | 学习率通常更小 |
| 稀疏特征模型 | AdaGrad / Adam | 坐标级自适应有利 |
| 强噪声强化学习 | Adam / RMSProp | RMSProp 在早期 RL 中常见 |
| 内存敏感场景 | Lion / SGD | 状态量更少 |

### 5.2 AdamW 常用超参起点

常见起点：

$$
\beta_1=0.9,
\quad
\beta_2=0.999,
\quad
\epsilon=10^{-8},
$$

权重衰减常在 $0.01$ 到 $0.1$ 间搜索，学习率需要按模型规模、batch size 和任务重新调。

### 5.3 哪些参数不做 weight decay

通常不对以下参数做权重衰减：

- bias；
- LayerNorm / BatchNorm 的 scale 和 shift；
- embedding 的某些特殊参数需按任务决定。

原因是这些参数不承担同样的容量控制角色，衰减可能损害训练稳定性。

### 5.4 梯度裁剪

全局范数裁剪：

$$
g \leftarrow g\cdot \min\left(1,\frac{c}{\|g\|_2}\right).
$$

常用于 RNN、Transformer、大模型训练，防止偶发梯度爆炸导致参数破坏。

### 5.5 Mixed precision 与损失缩放

FP16 / BF16 可提升吞吐，但 FP16 动态范围较小。常见策略：

1. 前向和反向使用低精度；
2. master weights 使用 FP32；
3. 对 loss 做 scale 后反传；
4. unscale 梯度后裁剪和 optimizer step。

BF16 动态范围更大，通常比 FP16 更少需要 loss scaling。

### 5.6 分布式训练中的优化器状态

AdamW 每个参数需要保存：

- 参数本身；
- 梯度；
- 一阶矩；
- 二阶矩；
- 可能还有 FP32 master copy。

这会带来数倍于参数量的显存开销。ZeRO、FSDP 等方法通过切分参数、梯度和优化器状态降低单卡内存压力。

### 5.7 常见踩坑清单

| 现象 | 可能原因 | 解法 |
| --- | --- | --- |
| loss 直接 NaN | 学习率过大、溢出、未裁剪 | 降学习率、启用 clipping、检查数据 |
| 训练初期震荡 | 无 warmup 或 batch 太大 | 加 warmup、降低峰值 lr |
| 验证集变差 | weight decay 不合适或过拟合 | 搜索 decay、增强数据、早停 |
| AdamW 不如 SGD | 任务泛化偏好不同 | 尝试 SGD + momentum 或更长训练 |
| 微调灾难性遗忘 | lr 过大、全量更新 | 降 lr、冻结部分层、LoRA |
| 梯度全零 | 激活饱和、detach 错误 | 检查计算图和激活分布 |
| 有效 lr 异常 | Adam 分母太小或 epsilon 不合适 | 观察 update norm，调 epsilon / lr |

### 5.8 监控指标

除了 loss，应监控：

- gradient norm；
- parameter norm；
- update-to-weight ratio：$\|\Delta\theta\|/\|\theta\|$；
- learning rate；
- Adam 的 $\sqrt{v_t}$ 分布；
- NaN / Inf 计数；
- clipping 触发比例。

这些指标能帮助判断是优化问题、数据问题还是数值问题。

---

## 06 统一对比框架

| 优化器 | 一阶矩 | 二阶矩 | 偏差修正 | 额外状态 | 典型默认 lr | 主要场景 |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 否 | 否 | 否 | 无 | $10^{-1}$ 到 $10^{-2}$ | 简单稳定、大规模视觉 |
| Momentum | 是 | 否 | 否 | 1 倍参数 | $10^{-1}$ 到 $10^{-2}$ | CNN、病态曲率 |
| Nesterov | 是 | 否 | 否 | 1 倍参数 | $10^{-1}$ 到 $10^{-2}$ | 需要更强前瞻的 SGD 变体 |
| AdaGrad | 否 | 累积平方 | 否 | 1 倍参数 | $10^{-2}$ | 稀疏特征 |
| RMSProp | 否 | EMA 平方 | 通常否 | 1 倍参数 | $10^{-3}$ | 非平稳 / RL |
| Adam | 是 | 是 | 是 | 2 倍参数 | $10^{-3}$ | 通用深度学习 |
| AdamW | 是 | 是 | 是 | 2 倍参数 | $10^{-4}$ 到 $10^{-3}$ | Transformer / LLM |
| AMSGrad | 是 | 最大二阶矩 | 是 | 2 倍参数 | $10^{-3}$ | 需要收敛修正的 Adam |
| Lion | 是 | 否 | 否 | 1 倍参数 | 常低于 AdamW | 内存敏感、需重调参 |

---

## 07 速记总结

> **SGD 直接沿随机负梯度走；Momentum 平滑方向；RMSProp / Adam 用历史平方梯度做坐标级缩放；AdamW 把权重衰减从梯度缩放中解耦。实际训练成败通常由学习率、warmup、weight decay、梯度裁剪和数值精度共同决定。**

---

## 附录: 极简 Python 代码

```python
import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.mu = momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.mu * self.v[i] + g
            p -= self.lr * self.v[i]


class AdamW:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # decoupled weight decay
            p *= (1 - self.lr * self.weight_decay)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm(grads, max_norm, eps=1e-12):
    total = 0.0
    for g in grads:
        total += np.sum(g * g)
    total_norm = np.sqrt(total)
    scale = min(1.0, max_norm / (total_norm + eps))
    return [g * scale for g in grads], total_norm


def cosine_lr(step, total_steps, base_lr, min_lr=0.0, warmup_steps=0):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine
```

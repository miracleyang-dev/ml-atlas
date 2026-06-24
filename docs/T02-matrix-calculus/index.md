# 【T02】Matrix Calculus 矩阵求导

---

## 00 前置记号与基础结论

### 记号约定

| 符号 | 含义 |
| --- | --- |
| $x \in \mathbb{R}^n$ | 列向量，默认所有向量均为列向量 |
| $X \in \mathbb{R}^{m \times n}$ | 矩阵，$X_{ij}$ 表示第 $i$ 行第 $j$ 列元素 |
| $f: \mathbb{R}^n \to \mathbb{R}$ | 标量值函数，梯度 $\nabla_x f \in \mathbb{R}^n$ |
| $g: \mathbb{R}^n \to \mathbb{R}^m$ | 向量值函数，Jacobian $J_g \in \mathbb{R}^{m \times n}$ |
| $df$ | 标量函数的一阶微分 |
| $dX$ | 矩阵变量的微小扰动 |
| $\mathrm{tr}(A)$ | 矩阵迹，主对角线元素之和 |
| $\langle A,B\rangle_F$ | Frobenius 内积，$\mathrm{tr}(A^\top B)$ |
| $\mathrm{vec}(X)$ | 按列堆叠矩阵得到的向量 |
| $A \otimes B$ | Kronecker 积 |
| $\odot$ | Hadamard 逐元素乘法 |

本文统一采用 **分子布局 / numerator layout**：

$$
\nabla_x f = \frac{\partial f}{\partial x} \in \mathbb{R}^{n},
\qquad
J_g = \frac{\partial g}{\partial x} \in \mathbb{R}^{m \times n},
\qquad
(J_g)_{ij}=\frac{\partial g_i}{\partial x_j}.
$$

在这个约定下，链式法则写成最符合反向传播直觉的形式：

$$
\nabla_x f(g(x)) = J_g(x)^\top \nabla_g f.
$$

### 必备前置定理

**迹循环性质**：只要维度相容，

$$
\mathrm{tr}(ABC)=\mathrm{tr}(BCA)=\mathrm{tr}(CAB).
$$

它的作用是把含有 $dX$ 的项移动到迹表达式最后，从而读出梯度。

**Frobenius 内积定义梯度**：对标量函数 $f(X)$，若一阶微分可写成

$$
df = \langle G, dX\rangle_F = \mathrm{tr}(G^\top dX),
$$

则 $G = \nabla_X f$。

**常用微分规则**：

$$
d(AB)=dA\,B + A\,dB, \qquad d(X^{-1})=-X^{-1}(dX)X^{-1}.
$$

**vec 恒等式**：

$$
\mathrm{vec}(AXB) = (B^\top \otimes A)\mathrm{vec}(X).
$$

它常用于把矩阵方程改写为标准线性代数问题，但工程上通常不显式构造 Kronecker 大矩阵。

---

## 01 问题定义与模型设定

矩阵求导的核心任务是：**给定一个由向量、矩阵、标量组合成的函数，得到形状正确、可用于优化或反向传播的梯度 / Jacobian / Hessian。**

### 1.1 标量对向量：最常见的训练目标

设模型参数 $\theta \in \mathbb{R}^p$，损失函数为

$$
L(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i), y_i).
$$

训练时需要计算

$$
\nabla_\theta L \in \mathbb{R}^{p},
$$

并用优化器更新 $\theta$。深度学习框架中的 `.backward()` 本质上就是高效计算这个梯度。

### 1.2 向量对向量：局部线性化

对 $g(x):\mathbb{R}^n\to\mathbb{R}^m$，一阶近似为

$$
g(x+\Delta x) \approx g(x) + J_g(x)\Delta x.
$$

Jacobian 描述输入扰动如何被函数映射到输出扰动。反向传播通常不显式形成完整 $J_g$，而是计算向量-Jacobian 积：

$$
J_g(x)^\top v.
$$

### 1.3 标量对矩阵：机器学习里的高频形态

线性回归、注意力层、低秩分解都会出现矩阵变量。例如

$$
F(W)=\frac{1}{2}\|XW-Y\|_F^2,
\qquad
X\in\mathbb{R}^{n\times d},
W\in\mathbb{R}^{d\times k},
Y\in\mathbb{R}^{n\times k}.
$$

目标是得到与 $W$ 同形状的梯度：

$$
\nabla_W F = X^\top(XW-Y).
$$

### 1.4 布局约定为什么重要

不同教材会用分子布局或分母布局。两者只是转置关系，但混用会导致链式法则方向错误。

| 对象 | 分子布局 | 分母布局 |
| --- | --- | --- |
| 标量 $f$ 对向量 $x$ | 列向量 $\nabla_x f$ | 行向量 $\partial f / \partial x$ |
| 向量 $g$ 对向量 $x$ | $m\times n$ | $n\times m$ |
| 链式法则 | $J_g^\top \nabla_g f$ | $\partial f/\partial g \cdot \partial g/\partial x$ |

本文后续全部使用分子布局。

---

## 02 完整数学推导

### 2.1 微分法：最稳定的手推路线

以二次型为例：

$$
f(x)=x^\top A x.
$$

一阶微分为

$$
\begin{aligned}
df
&= d(x^\top A x) \\
&= (dx)^\top A x + x^\top A(dx) \\
&= x^\top A^\top dx + x^\top A dx \\
&= \big((A+A^\top)x\big)^\top dx.
\end{aligned}
$$

因此

$$
\boxed{\nabla_x (x^\top A x) = (A+A^\top)x.}
$$

若 $A=A^\top$，则 $\nabla_x f=2Ax$。

### 2.2 迹技巧：从矩阵微分读出梯度

考虑

$$
f(X)=\mathrm{tr}(A^\top X), \qquad A,X\in\mathbb{R}^{m\times n}.
$$

有

$$
df=\mathrm{tr}(A^\top dX)=\langle A,dX\rangle_F,
$$

所以

$$
\boxed{\nabla_X \mathrm{tr}(A^\top X)=A.}
$$

再看线性变换后的迹：

$$
f(X)=\mathrm{tr}(A X B).
$$

微分为

$$
df=\mathrm{tr}(A\,dX\,B)=\mathrm{tr}(BA\,dX)=\mathrm{tr}((A^\top B^\top)^\top dX),
$$

因此

$$
\boxed{\nabla_X \mathrm{tr}(A X B)=A^\top B^\top.}
$$

关键不是背公式，而是把表达式整理成 $\mathrm{tr}(G^\top dX)$。

### 2.3 最小二乘梯度与正规方程

令

$$
F(W)=\frac{1}{2}\|XW-Y\|_F^2.
$$

记残差 $E=XW-Y$，则

$$
F(W)=\frac{1}{2}\mathrm{tr}(E^\top E).
$$

微分：

$$
\begin{aligned}
dF
&=\frac{1}{2}\mathrm{tr}(dE^\top E + E^\top dE) \\
&=\mathrm{tr}(E^\top dE) \\
&=\mathrm{tr}(E^\top X\,dW) \\
&=\mathrm{tr}((X^\top E)^\top dW).
\end{aligned}
$$

所以

$$
\boxed{\nabla_W F = X^\top(XW-Y).}
$$

令梯度为零得到正规方程：

$$
X^\top X W = X^\top Y.
$$

若 $X^\top X$ 可逆，则

$$
W=(X^\top X)^{-1}X^\top Y.
$$

工程上不推荐显式求逆，应使用 QR、SVD 或线性方程求解。

### 2.4 Logistic 回归的梯度

二分类 Logistic 回归：

$$
z=Xw+b\mathbf{1}, \qquad p=\sigma(z), \qquad \sigma(t)=\frac{1}{1+e^{-t}}.
$$

平均交叉熵：

$$
L(w,b)=-\frac{1}{n}\sum_{i=1}^n\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

单样本有

$$
\frac{\partial \ell}{\partial z}=p-y.
$$

批量形式：

$$
\boxed{\nabla_w L=\frac{1}{n}X^\top(p-y),\qquad
\nabla_b L=\frac{1}{n}\mathbf{1}^\top(p-y).}
$$

这也是神经网络最后一层 sigmoid + binary cross entropy 的标准反传结果。

### 2.5 Softmax + Cross Entropy 的核心化简

多分类 logits $z\in\mathbb{R}^C$：

$$
p_j=\frac{e^{z_j}}{\sum_{k=1}^C e^{z_k}}.
$$

交叉熵：

$$
\ell(z,y)=-\sum_{j=1}^C y_j\log p_j.
$$

softmax 的 Jacobian 为

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
$$

与交叉熵组合后大幅简化：

$$
\boxed{\nabla_z \ell = p-y.}
$$

若最后一层为 $Z=HW+b$，其中 $H\in\mathbb{R}^{n\times d}$，$W\in\mathbb{R}^{d\times C}$，则

$$
\boxed{\nabla_W L=\frac{1}{n}H^\top(P-Y),\quad
\nabla_H L=\frac{1}{n}(P-Y)W^\top,\quad
\nabla_b L=\frac{1}{n}\mathbf{1}^\top(P-Y).}
$$

### 2.6 两层网络反向传播

设

$$
H=\phi(A),\qquad A=XW_1+b_1,
$$

$$
Z=HW_2+b_2,
\qquad
L=\mathrm{CE}(\mathrm{softmax}(Z),Y).
$$

从输出层开始：

$$
G_Z=\frac{1}{n}(P-Y).
$$

然后逐层向后：

$$
\nabla_{W_2}L=H^\top G_Z,
\qquad
\nabla_{b_2}L=\mathbf{1}^\top G_Z,
$$

$$
G_H=G_ZW_2^\top,
\qquad
G_A=G_H\odot \phi'(A),
$$

$$
\boxed{\nabla_{W_1}L=X^\top G_A,
\qquad
\nabla_{b_1}L=\mathbf{1}^\top G_A.}
$$

反向传播不是另一套数学，而是链式法则在计算图上的高效组织。

### 2.7 Hessian 与二阶近似

对 $f:\mathbb{R}^n\to\mathbb{R}$，Hessian 为

$$
H_f(x)=\nabla_x^2 f \in \mathbb{R}^{n\times n},
\qquad
(H_f)_{ij}=\frac{\partial^2 f}{\partial x_i\partial x_j}.
$$

二阶 Taylor 展开：

$$
f(x+\Delta x)\approx f(x)+\nabla f(x)^\top\Delta x+\frac{1}{2}\Delta x^\top H_f(x)\Delta x.
$$

对最小二乘

$$
f(w)=\frac{1}{2}\|Xw-y\|_2^2,
$$

梯度和 Hessian 为

$$
\nabla_w f=X^\top(Xw-y),\qquad \nabla_w^2 f=X^\top X.
$$

若 $X^\top X$ 正定，Newton 步为

$$
w_{t+1}=w_t-(X^\top X)^{-1}X^\top(Xw_t-y).
$$

二阶方法利用曲率，但大模型中完整 Hessian 通常不可存储，只能用 Hessian-vector product 或近似二阶方法。

---

## 03 几何直观解释

### 梯度：最速上升方向

对小扰动 $\Delta x$，一阶变化为

$$
f(x+\Delta x)-f(x)\approx \nabla f(x)^\top \Delta x.
$$

在 $\|\Delta x\|_2$ 固定时，内积最大方向是 $\Delta x \parallel \nabla f(x)$。因此负梯度方向是最速下降方向。

### Jacobian：局部线性变换

Jacobian 把输入空间中的小向量映射到输出空间：

$$
\Delta y \approx J\Delta x.
$$

如果 $J$ 的某个奇异值很大，说明该方向上的输入扰动会被放大；如果奇异值很小，说明该方向被压扁。这也是梯度爆炸 / 消失的一种线性化解释。

### Hessian：局部曲率

Hessian 的特征值描述曲率：

- 正特征值：该方向局部向上弯曲；
- 负特征值：该方向局部向下弯曲；
- 接近零：该方向平坦，优化器移动缓慢。

深度网络的损失面常有大量近零曲率方向，所以一阶优化器虽简单，却在大规模训练中更稳健。

---

## 04 核心性质与关键定理

### 4.1 常见求导公式表

| 表达式 | 梯度 |
| --- | --- |
| $a^\top x$ | $a$ |
| $x^\top A x$ | $(A+A^\top)x$ |
| $\frac{1}{2}\|Ax-b\|_2^2$ | $A^\top(Ax-b)$ |
| $\mathrm{tr}(A^\top X)$ | $A$ |
| $\mathrm{tr}(AXB)$ | $A^\top B^\top$ |
| $\frac{1}{2}\|XW-Y\|_F^2$ | $X^\top(XW-Y)$ |
| $\log\det X$ | $X^{-\top}$ |
| $\mathrm{tr}(X^{-1}A)$ | $-X^{-\top}A^\top X^{-\top}$ |

### 4.2 链式法则的形状检查

若

$$
x\in\mathbb{R}^n,\quad u=g(x)\in\mathbb{R}^m,\quad f=h(u)\in\mathbb{R},
$$

则

$$
\nabla_x f = J_g^\top \nabla_u h.
$$

形状为

$$
(n\times m)(m\times 1)=n\times 1.
$$

如果推导中出现 $J_g\nabla_u h$，通常说明布局混乱。

### 4.3 VJP 与 JVP

深度学习框架最常用的是 VJP：

$$
\mathrm{VJP}(v)=J^\top v.
$$

它适合标量损失对大量参数求梯度，因为从输出端一个标量反传即可。

JVP 为

$$
\mathrm{JVP}(u)=Ju.
$$

它适合前向敏感性分析、Neural ODE、Hessian-vector product 等场景。

### 4.4 Hessian-vector product

显式 Hessian 需要 $O(p^2)$ 存储。Hessian-vector product 可由梯度再求导得到：

$$
Hv=\nabla_x\left(\nabla_x f(x)^\top v\right).
$$

这让共轭梯度、Lanczos 谱估计、影响函数等方法无需形成完整 Hessian。

### 4.5 对称性与 PSD

若 $f$ 二阶连续可微，则 Hessian 对称：

$$
\frac{\partial^2 f}{\partial x_i\partial x_j}=\frac{\partial^2 f}{\partial x_j\partial x_i}.
$$

若 $f$ 为凸函数，则

$$
\nabla^2 f(x)\succeq 0.
$$

最小二乘的 Hessian $X^\top X$ 总是半正定，这解释了其凸性。

---

## 05 工程实战要点

### 5.1 优先用微分推导，最后再转成代码

推荐流程：

1. 写出标量目标；
2. 求一阶微分；
3. 整理成内积形式；
4. 读出梯度；
5. 做形状检查；
6. 用有限差分或 autograd 校验。

### 5.2 避免显式求逆

表达式

$$
w=(X^\top X)^{-1}X^\top y
$$

适合数学说明，不适合直接实现。更稳定的写法是解线性方程：

$$
(X^\top X)w=X^\top y.
$$

若矩阵病态，优先使用 QR、SVD 或加 Ridge 正则。

### 5.3 数值稳定的 softmax

不要直接计算 $e^{z_i}$，应先减最大值：

$$
\mathrm{softmax}(z)_i=\frac{e^{z_i-\max_j z_j}}{\sum_k e^{z_k-\max_j z_j}}.
$$

这不改变结果，但能防止指数溢出。

### 5.4 批量维度约定

工程中最常见的错误不是公式错，而是 batch 维度放错。本文默认：

- 数据矩阵 $X$ 每行一个样本；
- 激活 $H$ 也是每行一个样本；
- 线性层 $Z=XW+b$；
- 权重梯度总是输入转置乘输出梯度。

### 5.5 常见踩坑清单

| 现象 | 原因 | 解法 |
| --- | --- | --- |
| 梯度形状转置 | 混用分子 / 分母布局 | 固定一种布局并做形状检查 |
| loss 变成 NaN | softmax / log 溢出 | 使用 log-sum-exp 技巧 |
| 手推与 autograd 差一倍 | 忘记 $\frac{1}{2}$ 或 batch 平均 | 对齐损失定义 |
| 矩阵求逆不稳定 | 条件数过大 | solve / QR / SVD / 正则化 |
| 广播导致静默错误 | bias 或 batch 维度不一致 | 明确 keepdims 与维度注释 |
| 梯度校验失败 | 有随机性或非光滑点 | 固定随机种子，避开 ReLU 零点 |

---

## 06 统一对比框架

| 求导对象 | 结果形状 | 典型工具 | 典型应用 |
| --- | --- | --- | --- |
| 标量对标量 | 标量 | 普通微积分 | 一维优化 |
| 标量对向量 | 向量 | 梯度 | 参数更新 |
| 向量对向量 | 矩阵 | Jacobian | 局部线性化、敏感性分析 |
| 标量对矩阵 | 同矩阵形状 | Frobenius 内积 | 线性层、矩阵分解 |
| 标量对向量二阶 | 方阵 | Hessian | Newton 法、曲率分析 |
| Jacobian 转置乘向量 | 向量 | VJP | 反向传播 |
| Jacobian 乘向量 | 向量 | JVP | 前向敏感性、HVP |

---

## 07 速记总结

> **矩阵求导的核心不是背公式，而是用微分把一阶变化写成内积形式；形状检查决定推导是否可信，链式法则决定反向传播如何高效实现。**

---

## 附录: 极简 Python 代码

```python
import numpy as np


def finite_diff_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


# ---------- least squares ----------
def least_squares_loss_grad(X, W, Y):
    E = X @ W - Y
    loss = 0.5 * np.sum(E * E)
    grad = X.T @ E
    return loss, grad


# ---------- stable softmax + cross entropy ----------
def softmax_cross_entropy_logits(Z, Y):
    Zs = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Zs)
    P = expZ / expZ.sum(axis=1, keepdims=True)
    n = Z.shape[0]
    loss = -np.sum(Y * np.log(P + 1e-12)) / n
    G = (P - Y) / n
    return loss, G


# ---------- two-layer network backward ----------
def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def two_layer_backward(X, Y, W1, b1, W2, b2):
    A = X @ W1 + b1
    H = relu(A)
    Z = H @ W2 + b2
    loss, Gz = softmax_cross_entropy_logits(Z, Y)

    dW2 = H.T @ Gz
    db2 = Gz.sum(axis=0, keepdims=True)
    dH = Gz @ W2.T
    dA = dH * relu_grad(A)
    dW1 = X.T @ dA
    db1 = dA.sum(axis=0, keepdims=True)
    return loss, (dW1, db1, dW2, db2)
```

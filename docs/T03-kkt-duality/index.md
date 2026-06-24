# 【T03】KKT & Duality 对偶问题

---

## 00 前置记号与基础结论

### 记号约定

| 符号 | 含义 |
| --- | --- |
| $x \in \mathbb{R}^n$ | 原问题决策变量 |
| $f_0(x)$ | 原问题目标函数 |
| $f_i(x) \le 0$ | 第 $i$ 个不等式约束 |
| $h_j(x)=0$ | 第 $j$ 个等式约束 |
| $\lambda_i \ge 0$ | 不等式约束的拉格朗日乘子 |
| $\nu_j \in \mathbb{R}$ | 等式约束的拉格朗日乘子 |
| $\mathcal{L}(x,\lambda,\nu)$ | 拉格朗日函数 |
| $g(\lambda,\nu)$ | 对偶函数 |
| $p^\star$ | 原问题最优值 |
| $d^\star$ | 对偶问题最优值 |
| $\mathcal{D}$ | 对偶可行域 |
| $\succeq 0$ | 半正定矩阵序 |

标准原问题写作：

$$
\begin{aligned}
\min_x \quad & f_0(x) \\
\text{s.t.}\quad & f_i(x) \le 0,\quad i=1,\dots,m, \\
& h_j(x)=0,\quad j=1,\dots,p.
\end{aligned}
$$

拉格朗日函数：

$$
\mathcal{L}(x,\lambda,\nu)=f_0(x)+\sum_{i=1}^m\lambda_i f_i(x)+\sum_{j=1}^p\nu_j h_j(x),
\qquad \lambda\ge 0.
$$

### 必备前置定理

**弱对偶**：对任意对偶可行 $(\lambda,\nu)$，都有

$$
g(\lambda,\nu) \le p^\star.
$$

因此对偶问题给出原问题最优值的下界。

**强对偶**：若原问题为凸优化问题，并满足适当正则条件（常见为 Slater 条件），则

$$
d^\star = p^\star.
$$

**KKT 条件**：在凸问题且强对偶成立时，KKT 条件是全局最优的充要条件。

---

## 01 问题定义与模型设定

KKT 与对偶理论解决的问题是：**如何把带约束优化问题转化为可分析、可求解、可解释的形式。**

### 1.1 约束优化的基本结构

无约束优化只需考虑梯度为零：

$$
\nabla f_0(x^\star)=0.
$$

有约束时，最优点可能出现在边界，梯度不一定为零。例如

$$
\min_x x^2 \quad \text{s.t.}\quad x\ge 1.
$$

最优解是 $x^\star=1$，但 $\nabla x^2=2\ne 0$。约束边界阻止继续沿负梯度方向移动。

### 1.2 不等式约束的标准形式

通常把约束写成

$$
f_i(x)\le 0.
$$

例如 $x\ge 1$ 写成

$$
1-x\le 0.
$$

乘子 $\lambda_i\ge 0$ 的符号来自弱对偶：对任意可行点，$\lambda_i f_i(x)\le 0$，从而拉格朗日函数在可行域上不超过或不低估目标，具体取决于最小化 / 最大化约定。本文固定为最小化标准形式。

### 1.3 对偶的直观目标

对偶函数定义为

$$
g(\lambda,\nu)=\inf_x \mathcal{L}(x,\lambda,\nu).
$$

它把 $x$ 消掉，只剩乘子。对偶问题为

$$
\max_{\lambda,\nu}\quad g(\lambda,\nu)
\quad \text{s.t.}\quad \lambda\ge 0.
$$

直观上，对偶是在寻找最紧的下界。

### 1.4 机器学习中为什么需要对偶

| 场景 | 对偶带来的价值 |
| --- | --- |
| SVM | 核技巧只能自然地出现在对偶形式中 |
| Lasso / 稀疏优化 | 对偶间隙可作为停止准则 |
| 最大熵模型 | 约束期望转化为指数族形式 |
| 资源分配 | 乘子可解释为影子价格 |
| 大规模优化 | 原变量和对偶变量维度不同，可选择更小的一侧求解 |

---

## 02 完整数学推导

### 2.1 从拉格朗日函数到弱对偶

对任意原问题可行点 $x$，有

$$
f_i(x)\le 0,
\qquad
h_j(x)=0.
$$

若 $\lambda_i\ge 0$，则

$$
\sum_i \lambda_i f_i(x)\le 0,
\qquad
\sum_j \nu_j h_j(x)=0.
$$

因此

$$
\mathcal{L}(x,\lambda,\nu)
= f_0(x)+\sum_i\lambda_i f_i(x)+\sum_j\nu_j h_j(x)
\le f_0(x).
$$

又因为

$$
g(\lambda,
\nu)=\inf_z \mathcal{L}(z,\lambda,\nu)
\le \mathcal{L}(x,\lambda,\nu),
$$

所以对任意可行 $x$，

$$
g(\lambda,\nu)\le f_0(x).
$$

特别地，取 $x=x^\star$ 得

$$
\boxed{g(\lambda,\nu)\le p^\star.}
$$

这就是弱对偶。

### 2.2 KKT 四条件

若 $x^\star,\lambda^\star,\nu^\star$ 为最优原对偶解，KKT 条件为：

**原始可行性**：

$$
f_i(x^\star)
\le 0,
\qquad
h_j(x^\star)=0.
$$

**对偶可行性**：

$$
\lambda_i^\star\ge 0.
$$

**互补松弛**：

$$
\lambda_i^\star f_i(x^\star)=0,
\qquad i=1,
\dots,m.
$$

**驻点条件**：

$$
\nabla_x \mathcal{L}(x^\star,\lambda^\star,\nu^\star)=0.
$$

互补松弛说明：每个不等式约束要么不活跃，要么乘子为正。

| 约束状态 | $f_i(x^\star)$ | $\lambda_i^\star$ | 含义 |
| --- | --- | --- | --- |
| 非活跃 | $<0$ | $0$ | 约束不影响最优解 |
| 活跃 | $=0$ | $\ge 0$ | 最优点贴在边界上 |

### 2.3 一维例子：边界最优

问题：

$$
\min_x x^2 \quad \text{s.t.}\quad 1-x\le 0.
$$

拉格朗日函数：

$$
\mathcal{L}(x,\lambda)=x^2+\lambda(1-x),
\qquad \lambda\ge 0.
$$

KKT 条件：

$$
1-x\le 0,
\qquad
\lambda\ge 0,
\qquad
\lambda(1-x)=0,
$$

$$
\frac{\partial \mathcal{L}}{\partial x}=2x-\lambda=0.
$$

若约束非活跃，则 $\lambda=0$，驻点给出 $x=0$，但不可行。故约束活跃：

$$
x=1,
\qquad
\lambda=2.
$$

这说明乘子大小刻画了边界约束对目标的“推力”。

### 2.4 等式约束例子：最小范数解

问题：

$$
\min_x \frac{1}{2}\|x\|_2^2
\quad \text{s.t.}\quad Ax=b.
$$

拉格朗日函数：

$$
\mathcal{L}(x,\nu)=\frac{1}{2}x^\top x+\nu^\top(Ax-b).
$$

驻点条件：

$$
\nabla_x\mathcal{L}=x+A^\top \nu=0
\quad\Rightarrow\quad
x=-A^\top\nu.
$$

代回约束：

$$
A(-A^\top\nu)=b
\quad\Rightarrow\quad
AA^\top\nu=-b.
$$

若 $AA^\top$ 可逆，

$$
\nu=-(AA^\top)^{-1}b,
\qquad
\boxed{x^\star=A^\top(AA^\top)^{-1}b.}
$$

这正是满足 $Ax=b$ 的最小二范数解。

### 2.5 二次规划的 KKT 线性系统

等式约束二次规划：

$$
\min_x \frac{1}{2}x^\top Qx+c^\top x
\quad \text{s.t.}\quad Ax=b,
\qquad Q\succ 0.
$$

拉格朗日函数：

$$
\mathcal{L}(x,\nu)=\frac{1}{2}x^\top Qx+c^\top x+\nu^\top(Ax-b).
$$

驻点与可行性：

$$
Qx+c+A^\top\nu=0,
\qquad
Ax=b.
$$

写成块线性系统：

$$
\boxed{
\begin{bmatrix}
Q & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
x \\
\nu
\end{bmatrix}
=
\begin{bmatrix}
-c \\
b
\end{bmatrix}.}
$$

这类鞍点系统是很多优化求解器的核心。

### 2.6 SVM 对偶推导

硬间隔 SVM 原问题：

$$
\begin{aligned}
\min_{w,b}\quad & \frac{1}{2}\|w\|_2^2 \\
\text{s.t.}\quad & y_i(w^\top x_i+b)\ge 1,
\quad i=1,
\dots,n.
\end{aligned}
$$

写成标准不等式：

$$
1-y_i(w^\top x_i+b)\le 0.
$$

拉格朗日函数：

$$
\mathcal{L}(w,b,\alpha)=\frac{1}{2}\|w\|_2^2+
\sum_{i=1}^n \alpha_i\left(1-y_i(w^\top x_i+b)\right),
\qquad \alpha_i\ge 0.
$$

驻点条件：

$$
\frac{\partial \mathcal{L}}{\partial w}=w-\sum_i\alpha_i y_i x_i=0
\quad\Rightarrow\quad
w=\sum_i\alpha_i y_i x_i,
$$

$$
\frac{\partial \mathcal{L}}{\partial b}=-\sum_i\alpha_i y_i=0
\quad\Rightarrow\quad
\sum_i\alpha_i y_i=0.
$$

代回得到对偶问题：

$$
\boxed{
\begin{aligned}
\max_\alpha\quad & \sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^\top x_j \\
\text{s.t.}\quad & \alpha_i\ge 0, \\
& \sum_i \alpha_i y_i=0.
\end{aligned}}
$$

软间隔 SVM 会增加上界约束 $0\le\alpha_i\le C$。

核技巧来自内积替换：

$$
x_i^\top x_j \rightarrow K(x_i,x_j).
$$

### 2.7 对偶间隙

对偶间隙定义为

$$
\mathrm{gap}=p-d,
$$

其中 $p$ 是某个原可行点的目标值，$d$ 是某个对偶可行点的对偶值。由弱对偶，gap 总非负。

在数值优化中，若

$$
\mathrm{gap} \le \epsilon,
$$

则可认为已接近最优。这比只看梯度范数更适合带约束问题。

---

## 03 几何直观解释

### 等式约束：梯度被约束法向量张成

若只有等式约束 $h(x)=0$，最优点处可行方向 $d$ 满足

$$
\nabla h(x^\star)^\top d=0.
$$

若沿任何可行方向目标都不能下降，则

$$
\nabla f_0(x^\star)^\top d=0
$$

对所有切向方向成立。因此 $\nabla f_0(x^\star)$ 必须落在约束法向量空间中：

$$
\nabla f_0(x^\star)+\nu\nabla h(x^\star)=0.
$$

### 不等式约束：只有活跃边界产生力

非活跃约束距离边界还有余量，不会限制局部移动，因此乘子为零。活跃约束像墙面一样提供法向反作用，乘子大小表示该约束的重要性。

### 对偶：从下界逼近最优值

每组乘子都给出一个下界 $g(\lambda,\nu)$。对偶问题是在所有下界中选最高的那个。强对偶成立时，最高下界正好碰到原问题最优值。

---

## 04 核心性质与关键定理

### 4.1 凸问题中的充要性

若 $f_0,f_i$ 为凸函数，$h_j$ 为仿射函数，且强对偶成立，则任何满足 KKT 条件的点都是全局最优点。

非凸问题中，KKT 通常只是局部最优的必要条件，还需要额外二阶条件或全局分析。

### 4.2 Slater 条件

对凸问题，若存在严格可行点 $\tilde{x}$ 使

$$
f_i(\tilde{x})<0,
\qquad
h_j(\tilde{x})=0,
$$

则通常强对偶成立。Slater 条件是工程中判断“对偶能不能放心用”的重要依据。

### 4.3 互补松弛的筛选作用

互补松弛

$$
\lambda_i f_i(x)=0
$$

意味着活跃集决定解的结构。许多算法本质上是在猜测或更新活跃集：

- active-set QP；
- SMO 求解 SVM；
- Lasso 中的坐标下降；
- interior-point 方法中的中心路径。

### 4.4 乘子的敏感性解释

若把约束右端放宽一点，最优值的变化率常由乘子给出。例如

$$
f_i(x)\le u_i.
$$

对应乘子可解释为资源边际价值，也叫影子价格。资源分配、经济学和网络流问题中非常常见。

### 4.5 Fenchel 对偶的视角

许多机器学习模型可写为

$$
\min_x f(Ax)+g(x).
$$

Fenchel 对偶会引入共轭函数：

$$
f^\star(y)=\sup_z\{y^\top z-f(z)\}.
$$

虽然形式更抽象，但它能统一解释 Lasso、Logistic 回归、最大熵模型等问题的对偶结构。

---

## 05 工程实战要点

### 5.1 先判断问题类型

推导或建模前先问：

1. 目标是否凸？
2. 不等式约束是否凸？
3. 等式约束是否仿射？
4. 是否存在严格可行点？
5. 变量维度和约束数量哪个更大？

这些决定是否可以使用 KKT 作为充要条件，以及是否值得转对偶。

### 5.2 数值求解器选择

| 问题类型 | 常见求解器 / 方法 |
| --- | --- |
| 线性规划 LP | simplex、interior-point |
| 二次规划 QP | active-set、OSQP、interior-point |
| 二阶锥 SOCP | ECOS、MOSEK |
| 半正定规划 SDP | interior-point、first-order SDP 方法 |
| 大规模光滑凸优化 | primal-dual gradient、ADMM |
| SVM | SMO、liblinear、libsvm |

### 5.3 约束尺度影响求解稳定性

如果某些约束数量级很大，乘子会被迫缩放，KKT 系统可能病态。应尽量标准化变量和约束，使不同约束的数值尺度接近。

### 5.4 不要把所有 KKT 点都当全局最优

非凸问题中，KKT 点可能是局部最小、局部最大或鞍点。神经网络训练中的约束优化也常遇到非凸结构，应配合数值实验、二阶信息或多初值检查。

### 5.5 常见踩坑清单

| 现象 | 原因 | 解法 |
| --- | --- | --- |
| 乘子符号反了 | 不等式方向写错 | 统一写成 $f_i(x)\le0$ |
| KKT 解不可行 | 忘记检查原始可行性 | 四条件逐条验证 |
| 对偶值大于原问题值 | 符号或 max/min 约定错 | 回到弱对偶推导检查 |
| 以为强对偶一定成立 | 问题非凸或不满足正则条件 | 检查凸性与 Slater 条件 |
| SVM 核化失败 | 原问题中直接替换内积 | 先推到对偶再核化 |
| 求解器不收敛 | 约束病态或尺度差异大 | 标准化、正则化、放宽容差 |

---

## 06 统一对比框架

| 视角 | 原问题 | 对偶问题 | 关系 |
| --- | --- | --- | --- |
| 变量 | 原始决策变量 $x$ | 乘子 $\lambda,\nu$ | 维度可能差异很大 |
| 目标 | 最小化原始代价 | 最大化下界 | 弱对偶保证下界有效 |
| 约束 | 业务或模型约束 | 乘子非负等约束 | 对偶约束来自下确界有限性 |
| 最优值 | $p^\star$ | $d^\star$ | 总有 $d^\star\le p^\star$ |
| 强对偶 | 不一定 | 不一定 | 凸性 + 正则条件常可保证 |
| KKT 作用 | 描述候选最优解 | 连接原对偶解 | 凸强对偶下为充要条件 |
| 工程用途 | 直接建模 | 核化、下界、停止准则 | 选择更易求的一侧 |

---

## 07 速记总结

> **KKT 把“最优点不能沿可行方向下降”写成代数条件；对偶把约束惩罚的乘子当作变量，用最紧下界逼近原问题。凸问题中，强对偶让这两套描述精确重合。**

---

## 附录: 极简 Python 代码

```python
import numpy as np


# ---------- equality constrained QP ----------
def solve_equality_qp(Q, c, A, b):
    """
    min 0.5 x^T Q x + c^T x
    s.t. A x = b
    """
    n = Q.shape[0]
    m = A.shape[0]
    KKT = np.block([
        [Q, A.T],
        [A, np.zeros((m, m))]
    ])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(KKT, rhs)
    x = sol[:n]
    nu = sol[n:]
    return x, nu


# ---------- KKT residual checker ----------
def kkt_residual(f_grad, constraints, x, lambdas, tol=1e-8):
    """
    constraints: list of callables f_i(x) <= 0
    f_grad: gradient of objective
    This simplified checker handles inequality constraints whose
    gradients are supplied separately as constraint.grad.
    """
    primal = np.array([con(x) for con in constraints])
    dual_feasible = np.all(lambdas >= -tol)
    primal_feasible = np.all(primal <= tol)
    comp = np.max(np.abs(lambdas * primal)) if len(lambdas) else 0.0

    stationarity = f_grad(x).copy()
    for lam, con in zip(lambdas, constraints):
        stationarity += lam * con.grad(x)

    return {
        "primal_feasible": primal_feasible,
        "dual_feasible": dual_feasible,
        "max_complementarity": comp,
        "stationarity_norm": np.linalg.norm(stationarity),
    }


# ---------- hard-margin SVM dual objective pieces ----------
def svm_dual_objective(alpha, X, y):
    Y = y[:, None] * y[None, :]
    K = X @ X.T
    return alpha.sum() - 0.5 * alpha @ (Y * K) @ alpha


def svm_primal_from_dual(alpha, X, y, eps=1e-8):
    w = (alpha * y) @ X
    support = np.where(alpha > eps)[0]
    if len(support) == 0:
        raise ValueError("no support vectors")
    b_values = y[support] - X[support] @ w
    b = b_values.mean()
    return w, b, support
```

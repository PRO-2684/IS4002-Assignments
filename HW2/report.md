<div align="center">
<h1>第二次作业（强化学习）</h1>
</div>
<div align="center">
<p>xxx PB21000000<br>2024 年 x 月 x 日</p>
</div>

## 问题 1：热身（10 分）

### a. 计算（5 分）

| $i$ \\ $s$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
| ---------- | ---- | ---- | --- | --- | --- |
| $0$        | 0    | 0    | 0   | 0   | 0   |
| $1$        | 0    | 7.5  | -10 | 20  | 0   |
| $2$        | 0    | 2.5  | 3   | 16  | 0   |

### b. 计算（5 分）

| $s$ | $\mu(s)$ |
| --- | -------- |
| -1  | $a_2$    |
| 0   | $a_1$    |
| 1   | $a_1$    |

## 问题 2：Q-Learning（15 分）

### a. 回答问题（2 分）

MDP 的特点是后续路径只与当前状态有关，而与之前状态无关，故变量 $G_t$ 仅与当前状态有关。而根据 $v(s), q(s, a)$ 的表达式，可以看出它们也和 $t$ 无关。

### b. 计算（8 分）

$\eta=\gamma=1$ 下：

| $t$ \\ $\hat{q}(s, a)$ | $(0, a_1)$ | $(0, a_2)$ | $(1, a_1)$ | $(1, a_2)$ |
| ---------------------- | ---------- | ---------- | ---------- | ---------- |
| 0                      | 0          | 0          | 0          | 0          |
| 1                      | 1          | 0          | 0          | 0          |
| 2                      | 1          | 0          | 3          | 0          |
| 3                      | 1          | 2          | 3          | 0          |
| 4                      | 1          | 2          | 3          | 0          |

### c. 回答问题（5 分）

收敛的前提：固定的、有限的马尔科夫决策过程；遵循一定的更新规则；每个状态-动作对都能被无限次访问。Q-Learning 能收敛的原因在于通过不断迭代更新 Q 表来逼近最优 Q 值，每一次更新都是在根据当前知识对未来奖励的最佳猜测的基础上进行。随时间推移，这一猜测变得越来越精确，直到最终达到稳定的最优解。

## 问题 3：Gobang Programming（55 分）

### a. 回答问题（2 分）

-  状态 $s$ 为 $3\times3$ 的矩阵，每格有 $3$ 种可能的值，那么总共有 $3^{3*3}=19683$ 个状态；行动 $a$ 有 $2*3*3=18$ 种值；那么要存储完整的 $Q^*$ 表则要 $19683*18=354294$ 个键值对。
- 内存方面：考虑用 $2$-bit 保存状态内的每个值，$5$-bit 保存一个行动，$32$-bit 存储浮点数 $Q$ 值。那么每个键值对需要 $2*9+5+32=55$ bit，总共需要 $55*354294=19486170\text{ bit}\approx2.32\text{ GB}$ 的内存，这在如今的电脑上是可以满足的。
- 时间方面：每迭代一轮就需要 $354294$ 次更新，而我们取 $10000$ 次迭代，这里的时间开销可能让人难以接受。

### b. 代码填空（33 分）

```python
class Gobang(UtilGobang):
    ########################################################################################
    # Problem 1: Modelling MDP.
    ########################################################################################

    # You do not have to modify the value of self.board during this step. #

    def get_next_state(self, action: Tuple[int, int, int], noise: Tuple[int, int, int]) -> np.array:
        """
        ...
        """

        # BEGIN_YOUR_CODE (our solution is 3 line of code, but don't worry if you deviate from this)
        next_state = copy.deepcopy(self.board)
        if action is not None:
            black, x, y = action
            next_state[x][y] = black
        # END_YOUR_CODE

        if noise is not None:
            white, x_white, y_white = noise
            next_state[x_white][y_white] = white
        return next_state

    def sample_noise(self) -> Union[Tuple[int, int, int], None]:
        """
        ...
        """
        if self.action_space:
            # BEGIN_YOUR_CODE (our solution is 2 line of code, but don't worry if you deviate from this)
            x, y = random.choice(self.action_space)
            self.action_space.remove((x, y))
            # END_YOUR_CODE
            return 2, x, y
        else:
            return None

    def get_connection_and_reward(self, action: Tuple[int, int, int],
                                  noise: Tuple[int, int, int]) -> Tuple[int, int, int, int, float]:
        """
        ...
        """

        # BEGIN_YOUR_CODE (our solution is 4 line of code, but don't worry if you deviate from this)
        black_1, white_1 = self.count_max_connections(self.board)
        next_state = self.get_next_state(action, noise)
        black_2, white_2 = self.count_max_connections(next_state)
        reward = (black_2 ** 2 - white_2 ** 2) - (black_1 ** 2 - white_1 ** 2)
        # END_YOUR_CODE

        return black_1, white_1, black_2, white_2, reward

    ########################################################################################
    # Problem 2: Implement Q learning algorithms.
    ########################################################################################

    def sample_action_and_noise(self, eps: float) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        ...
        """

        # BEGIN_YOUR_CODE (our solution is 8 line of code, but don't worry if you deviate from this)
        choose_randomly = random.random() < eps
        random_action = (1,) + random.choice(self.action_space)
        if choose_randomly or len(self.Q) == 0:
            action = random_action
        else:
            s = self.array_to_hashable(self.board)
            action = max(self.Q[s], key=self.Q[s].get, default=random_action) # default to random action if no action in Q
        self.action_space.remove(action[1:])
        # END_YOUR_CODE
        return action, self.sample_noise()

    def q_learning_update(self, s0_: np.array, action: Tuple[int, int, int], s1_: np.array, reward: float,
                          alpha_0: float = 1):
        """
        ...
        """

        s0, s1 = self.array_to_hashable(s0_), self.array_to_hashable(s1_)
        self.s_a_visited[(s0, action)] = 1 if (s0, action) not in self.s_a_visited else \
            self.s_a_visited[(s0, action)] + 1
        alpha = alpha_0 / self.s_a_visited[(s0, action)]

        # BEGIN_YOUR_CODE (our solution is 18 line of code, but don't worry if you deviate from this)
        if s0 not in self.Q: # initialize Q[s0] if not exist
            self.Q[s0] = {}
        q = self.Q[s0].get(action, 0) # default to 0 if action not in Q[s0]
        if s1 not in self.Q:
            self.Q[s1] = {}
        max_q = max(self.Q[s1].values()) if self.Q[s1] else 0 # default to 0 if Q[s1] is empty
        q = alpha * (reward + self.gamma * max_q) + (1 - alpha) * q # new Q value
        self.Q[s0][action] = q # update Q[s0][action]

        return
        # END_YOUR_CODE
```

### c. 结果复现（10 分）

![训练结束](assets/Pasted%20image%2020240410230651.png)

![复现结果](assets/Pasted%20image%2020240410230735.png)

可以看到黑子胜率是 96.5%，效果还是很好的。

### d. 回答问题（10 分）

![](assets/Pasted%20image%2020240411151500.png)

$n=4$ 时胜率为 95.7%，与之前相比有小幅改变，但是基本相等。这符合预期，因为我们的 MDP 模型是通用的，不局限于 $n=3$ 的情况。

## 问题 4：Deeper Understanding（10 分）

### a. 回答问题（5 分）

$(\mathcal{T}_{\mu}v)(s)=r_{sa}+\gamma\cdot\sum_{s^{\prime}\in\mathcal{S}}p_{sas^{\prime}}\cdot v(s^{\prime})\quad(a=\mu(s))$

### b. 回答问题（5 分）

$$
\begin{aligned}
&|\mathcal\tau_{v_1}(s) - \mathcal\tau_{v_2}(s)| \\
&= \left|\mathbb E_{a\in\mathcal A}\left[r_{sa}+\gamma\cdot\sum_{s' \in\mathcal S}  p_{sas'}v_1(s')-\left(r_{sa}+\gamma\cdot\sum_{s' \in\mathcal S}  p_{sas'}v_2(s')\right)\right]\right| \\
&=\gamma\cdot\left|\mathbb E_{a\in\mathcal A}\left[\mathbb E_{s'\sim p_{sas'}}[v_1(s')-v_2(s')]\right]\right| \\
&=\gamma\cdot\left|\mathbb E_{a\in\mathcal A, s'\sim p_{sas'}}[v_1(s')-v_2(s')]\right|  \\
&\le\gamma\cdot \mathbb E_{a\in\mathcal A, s'\sim p_{sas'}}|v_1(s')-v_2(s')| \\
&\le\gamma\cdot \mathbb E_{a\in\mathcal A, s'\sim p_{sas'}}\max|v_1(s)-v_2(s)| \\
&=\gamma\cdot ||v_1-v_2||_\infty\quad(0\leq\gamma\leq1)
\end{aligned}
$$

那么我们有：$\forall s, ||\mathcal\tau_{v_1} - \mathcal\tau_{v_2}||_\infty = \max_s|\mathcal\tau_{v_1}(s) - \tau_{v_2}(s)| \leq \gamma\cdot |v_1-v_2|_\infty$。根据定义，可知 $\tau$ 为压缩映射。

## 反馈（10 分）

- 花费大概 6h
- 代码难度河里，证明题可以适当减少

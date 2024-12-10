
# 零、题记


**强化学习的知识很琐碎，我自己当初学的很吃力，特此整理博客，以便后人批判性学习………………**
这篇博客一方面为了记录当前所整理的知识点，另一方面PPO算法实在是太重要了，不但要从理论上理解它到底是怎样实现的，还需要从代码方面进行学习，这里我就通俗的将这个知识点进行简单的记录，用来日后自己的回顾和大家的交流学习。
下面均是我自己个人见解，如有不对之处，欢迎评论区指出错误！


# 一. 公式推导


这里简要交代PPO的算法原理及思想过程，主要记录自己的笔记，公式记录比较详细，我这里就不再赘述了，后面代码会紧紧贴合前面的内容，并且会再次提到一些细节。
![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123524340-180754122.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123425764-1676390147.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123443000-1314441125.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123448781-2130673015.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123452780-1183412724.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123457990-1691198743.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123502558-793951971.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123508049-1028146139.png)


![image](https://img2024.cnblogs.com/blog/3481742/202411/3481742-20241101123514891-1347579236.png)


![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209194840491-1736040983.png)


 好到这里就是PPO的基本思想和RL的前期铺垫工作了，这就是理论，脱离实践的理论永远也没办法好好理解，那么下面我们来看看代码部分。


# 二 代码公示


代码选择的是 [动手学强化学习](https://github.com "动手学强化学习")
下面我将逐一刨析整个代码的全部过程，之后大家可以将这些过程迁移到Issac Gym中进行学习和思考。


## 1 task\_PPO.py文件



点击查看代码

```
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

if __name__ == "__main__":
    #初始化参数
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cpu")
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    #初始化智能体
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    #开始训练
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

```


## 2 rl\_utils.py文件



点击查看代码

```
from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

```


## 3 开始刨析


### （1）先从主函数运行开始，其他的遇到了我们再来分析


![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209193932932-1081938741.png)
首先是对函数的一些变量进行初始化，都包括Adam优化器的学习率，episode的长度，隐藏层的维数，奖励函数的系数γγ值，GAE的λλ值，epoch的长度(本代码表示的是进行梯度反向传播的轮数)，eps表示PPO中截断范围clip函数的参数（\-eps\~\+eps），device是什么设备（源代码是cuda，这里大家可以更改下，因为我是用cpu跑的），然后下面的就是代码的一些初始化了，我这里不再赘述。
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209195633556-1790982838.png)
这哥俩就表述:前者是观测值的数量（比如读取的电机这时的角度、力矩、电机此时速度、机器人此时欧拉角、base线速度等【扯远了】），后者是动作空间的数量（比如输出的电机该咋转的角度，比如一只A1有12个电机，那么输出的action的维数就是12，代表着每一个电机最后应该转多少角度，往哪里转）


### （2）初始化智能体


![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209200010178-1826562222.png)


这时我们调用PPO函数，然后我们来看一下PPO函数具体都干了些什么。（这里就是起了这个名字，不要误会它在这个函数里把所有PPO算法都实现了）
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209200107294-1880625415.png)


我们下面一步一步来看。
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209200237050-703988568.png)


#### a.第一个地方


这里是对Actor\-Critic算法的复现，定义了两个神经网络，这两个神经网络具体长什么样子呢？我们来看看：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209200356339-116892157.png)


PolicyNet是Actor，相当于前面我们提到的Policy Improvement,简称PI，是策略改善的网络，最终我们希望输入一个状态，我们就知道该使用什么样子的action，黑盒明白吧。
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209201154364-829633520.png)


ValueNet是Critic，对Actor的动作做评估，进而更新state value或者action value的值。网络结构我也大致画一下：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209201519780-405157386.png)
其实也就相当于前面去掉了softmax，至于为什么呢？
你看哈，actor网络的作用是选择一个动作进行输出，所以我们使用softmax进行分类，理论上选择让累计奖励更大的动作作为较大概率的输出（greedy action)，所以这里用了一个分类器。


#### b.第二个地方


由于需要对两个网络进行反向传播，所以这里定义了两个Adam优化器进行。
普通咱一般用的是这种：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209201934503-1365921736.png)


Adam优化器是这样子：（[如果比较感兴趣的话，这篇博客个人感觉讲的比较好！）](https://github.com "如果比较感兴趣的话，这篇博客个人感觉讲的比较好！）")
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209201950701-1870277770.png)


代码里的lr就是咱前面提到的学习率，一般都是10的负几次方。


#### c.第三个地方就没啥说的了，几个赋值


记住以后再看torch的代码，python的代码，一般很喜欢用class的形式，这时你就把`self`看成不同函数之间变量的搬运工，只要有定义`self.xxxx`你就要知道一般这东西是在别处定义的，或者别处也能用得到的，这一点不是很绝对，但至少对初学者来说很实用。


### （3）开始训练


![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209202221084-1936316737.png)
我们转到对应的函数文件去看看（ctrl\+鼠标左键）
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209202408670-1598254941.png)
可以说这个函数是这个代码最核心的部分，也是PPO算法最核心的部分，我们来看看咋实现的。


#### a.第一话


![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209202605282-2115856473.png)
首先循环10次，每一轮咱训练50个episodes，一个episode就包含着很多个\<state,action\>\<state,action\>对（这个对长度不一定，可能随时停止，也可能尽可能探索到最大的episode\_max长度）,最后10x50等于500，所有episodes都执行完毕，此处对应代码：`num_episodes = 500`
然后是用tqdm定义一个进度条（这个不是重点，知道就行）
初始化变量transitiondicttransitiondict，这个变量包含5个值：现在的状态states,执行的动作actions，执行完到下一个时刻的状态next\_states，奖励函数rewards，以及这一时刻执行完的结束标志done(结束为1，否则为0，记住后面代码里要用到)


#### b.第二话


接着上面的代码再继续分析
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209203357867-8053495.png)


##### I)take actions


![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209203438252-1788661996.png)
可以看到首先将输入的state转换为tensor，然后因为actor网络的最后一层是softmax函数，所以通过actor网络输出两个执行两个动作可能性的大小，然后通过`action_dist = torch.distributions.Categorical(probs) action = action_dist.sample()`根据可能性大小进行采样最后得到这次选择动作1进行返回。


![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209203923094-967191751.png)


##### II)env.step(action)


按照采取的动作，和环境进行交互，返回到达下一时刻状态的值、奖励函数以及一个episode是否结束的结束标志done。


##### III)transition\_dist


然后将这些返回的值装进这个字典里面，有点像replaybufferreplaybuffer我个人感觉。
然后更新下一时刻的状态值，这个episode的奖励函数也将累加`episode_return += reward`
然后通过`return_list.append(episode_return)`记录下这一整个episode的总的累计奖励R,用来plt绘图吧，这个就类似TensorBoard的观测输出了。


##### IV)agent.update(...)


`agent.update(transition_dict)`到了这个函数。
这个函数说明了我们如何利用这些变量进行更新呢？
首先先从`transition_dict`取出变量存入`tsnsor`中便于计算
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209205502249-1321732564.png)


然后定义了TD Target：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209205621319-87782069.png)
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209205939868-1735929841.png)
然后是TD error
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209210002609-926595335.png)
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209210113899-44520640.png)
最后这俩就一起说了
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209210126250-759548909.png)
前一个是计算GAE
`advantage`表示的是某状态下采取某动作的优势值，也就是咱前面提到的PPO算法公式中的A：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209210209373-1534268096.png)
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209210340273-910673064.png)
`old_log_probs`则表示的是旧策略下某个状态下采取某个动作的概率值的对数值。对应下面PPO算法公式中的clip函数中分母:
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209210653553-978044262.png)
然后这个函数下面就老有意思了：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209210717034-1647613001.png)
首先第一部分： `log_probs = torch.log(self.actor(states).gather(1, actions))`
求对数值，然后下面新旧相减：
`ratio = torch.exp(log_probs - old_log_probs)`
这不就是e\[ln(a)−ln(b)]\=ab嘛
其实它就是在计算这个：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209211143312-484052691.png)


然后下面的surr1和surr2起初我也看不懂，后来看看算法自然而然就明白是干嘛的了：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209211227391-674344778.png)
这俩其实就是用前面的`ratio`分别相乘来作为下图的两个值，结果再求最小值，讲约束条件纳入公式里面，这就是PPO的核心！！！
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209211342675-23426775.png)
这其实就已经写完了actor的损失函数了，也就是策略的损失函数了。至于critic的损失函数，是在求td target和`critic(states)`的MSE,均方误差。
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209211439084-1733463788.png)
然后清零Adam，梯度反向传播，更新。。循环`epoch=10`次进行参数的神经网络参数的更新，每一次相当于收集了50个episodes。


#### c.



```
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

```

更新进度条，跟咱要学的PPO没关系。


#### d.


 `return return_list`返回这个奖励函数表。用来最后的输出评估：


### （4）记录


大家可能对整个流程比较糊涂，这里简单的做一个梳理：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209213346743-1559747173.png)


下面引用自其他佬的博客：
![image](https://img2024.cnblogs.com/blog/3481742/202412/3481742-20241209212856480-1321790467.png)


# 三、Acknowledge


感谢这些大佬的博客和思路分享，才让我渐渐理解整个代码思想。
[鸣谢1](https://github.com "主要鸣谢1"):[FlowerCloud机场订阅官网](https://hanlianfangzhi.com)
[鸣谢2](https://github.com "主要鸣谢2")
[鸣谢3](https://github.com "主要鸣谢3")



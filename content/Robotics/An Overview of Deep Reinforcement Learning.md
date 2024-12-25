---
draft: false
tags:
  - robotics
  - tutorial
date: 2024-06-22
---
 
# Intro to RL

## What is RL?

>[!important]
>- In RL, an **agent** learns to make decisions by interacting with an environment (through trial and error)
>- The agent receives feedback in the form of rewards. 
>- The goal is to **maximize** the total reward over time.

>[!question] How does the agent make decisions? 
>- The agent uses a **policy** ($\pi$) to *decide what action to take in a given state*.
>- Given a state, the policy outputs an action (if deterministic) or a probability distribution over actions (if stochastic).
> - The goal of RL is to find an **optimal policy** $\pi ^ *$ that maximizes the expected cumulative reward.

>[!question] What is the Exploration/Exploitation trade-off?
>- **Exploration**: trying out new actions to discover the best ones.
>- **Exploitation**: choosing the best actions based on what we already know.

> [!question] What are the two main types of RL algorithms?
>- **Value-based methods**: estimate the value of being in a state (or taking an action in a state) and use this to make decisions.
  >	- *train a value function* that estimates the expected cumulative reward of being in a state.
> - **Policy-based methods**: *train a policy function* that outputs the action to take in a given state.

### Value-based methods
> [!question] What is the value of a state?
> - The value of a state is the expected discounted return the agent can get if it starts at that state and then acts *according to our policy.*

>[!question] But what does it mean to "act according to our policy?"
>- For value-based methods, you don't train the policy. 
>- Instead, the policy is just a simple **pre-specified function** that uses the values given by the value-function to select its actions. 
  >- The greedy policy is an example of this: it selects the action that maximizes the value function.
    > - **Epsilon-greedy policy** is commonly used (it handles the exploration/exploitation trade-off that will be mentioned later). 

> [!tldr]
>* in policy-based methods, the optimal policy ($\pi ^ *$) is found by training the policy directly.
> * in value-based methods, the optimal policy is found by finding an optimal value function ($V^*$ or $Q^*$) and then using it to derive the optimal policy.

>[!question] What is the link between Value and Policy? 
> $$ \large \boxed{\pi ^ * (s) = \arg \max _a Q^* (s, a)}$$

>[!important] State-Value Function 
>- The state-value function ($V^{\pi} (s)$) estimates the expected cumulative reward of being in a state and following a policy $\pi$.
>- 
> $$\large V_{\pi} (s) = \mathbb{E} [R_t | s_t = s, \pi]$$
> 
> - For each state, the state-value function outputs the expected return if the agent starts at that state and then follows the policy $\pi$ forever. 

>[!important] The action-value function
> - The action-value function ($Q^{\pi} (s, a)$) estimates the expected cumulative reward of taking an action in a state and then following a policy $\pi$.
> $$\large Q_{\pi} (s, a) = \mathbb{E_\pi} [R_t | s_t = s, a_t = a]$$

>[!question] Action-Value VS. State-Value function:
> - **State-value function**: calculates the value of a state.
> - **Action-value function**: calculates the value of taking an action in a state (state-action pair). 

## The Bellman Equation

>[!question] What is the Bellman equation?
>A way to simplify our state-value or state-action value calculation. 

Remember: To calculate the value of a state, we need to calculate the return starting at that state and then following the policy forever. 

However, when calculating $V(S_{t})$, we need to know $V(S_{t+1})$.

Thus, to avoid repeated computation, we use Dynamic Programming, specifically, the Bellman Equation:

$$
\Large \boxed{V_\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma V_\pi(S_{t+1}) | S_{t} = s]}
$$

The value of our state is the expected reward we get at the next time step plus the (discounted) expected value of the next state.

The value of $V(S_{t+1})$ is equal to the immediate reward $R_{t+2}$ plus the discounted value of the next state ($\gamma \cdot V(S_{t+2})$).

## Monte Carlo and Temporal Difference Learning
### Monte Carlo: learning at the end of an episode
- In Monte Carlo Learning, you wait until the end of the epsiode, then calculate the return $G_t$ , and then update the value function $V(S_t)$.
  
$\Large \boxed{V(S_t) = V(S_t) + \alpha (G_t - V(S_t))}$

# Q-Learning

> [!summary] **What is Q-Learning?**
> Q-Learning is an **off-policy**, **value-based** method that uses a **TD** approach to train its action-value function. 
>> [!remember] Remember
>> - Recall: value-based method: find the optimal policy indirectly by training a value/action-value function that tells us the value of each state / value of each state-action pair 
>> - Recall: TD approach: update the value function after each time step, instead of at the end of the episode. 


> [!question] What is the difference between value and reward?  
> - **Reward**: The immediate reward received after taking an action in a state.  
> - **Value**: The expected cumulative reward from taking an action in a state and following a policy thereafter.  

---

> [!summary] Representing the Q-Function  
> - The **Q-function** is a table (Q-table) that maps state-action pairs to values.  
> - It tells us the value of taking a specific action in a given state.  
> - Initially, the Q-table is uninformative (all values are zero or random), but as the agent **explores the environment**, the Q-table is updated to approximate the optimal policy.  

> [!question] Why does Q-Learning allow us to learn the optimal policy?  
> - By training the Q-function, represented as a Q-table, we derive the optimal policy since it maps each state-action pair to the best action.  

---

> [!example] The Q-Learning Algorithm  
> - **Step 1**: **Initialize** the Q-table:  
>   > - Set $Q(s, a) = 0$ for all $s \in S$ and $a \in A(s)$.  
>   > - Ensure $Q(\text{terminal\_state}, \cdot) = 0$.  
> - **Step 2**: **Choose an action** using the epsilon-greedy strategy:  
>   > - Initialize $\epsilon = 1.0$.  
>   > - **Exploration**: With probability $\epsilon$, pick a random action.  
>   > - **Exploitation**: With probability $1 - \epsilon$, pick the best action from the Q-table.  
> - **Step 3**: **Perform the action** $A_t$:  
>   > - Observe reward $R_{t+1}$ and the next state $S_{t+1}$.  
> - **Step 4**: **Update the Q-value** using the Bellman equation:  
>   > $$ 
>   > \large Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right].
>   > $$

---
## DQN
> [!question] Why do we use Deep Q-Learning?  
> - The traditional Q-Learning **Q-table** becomes impractical for large state spaces.  
> - Deep Q-Learning replaces the Q-table with a neural network that **approximates the Q-function**, enabling efficient handling of large state spaces.  

---

> [!summary] Deep Q-Learning Architecture & Algorithm  
> - **Input**: A stack of 4 frames passed through a convolutional neural network (CNN).  
>   > [!question] Why 4 frames?  
>   > - To capture the motion of the agent.  
> - **Output**: A vector of Q-values, one for each possible action.  

> [!summary] Loss Function:  MSE
> - Minimize the **mean squared error** between predicted and target Q-values.  
> - Use the Bellman equation to calculate the target Q-values.  

> [!example] Training Deep Q-Learning  
> - **Step 1**: Sampling  
>   > - Perform actions in the environment.  
>   > - Store observed experience tuples $(S_t, A_t, R_{t+1}, S_{t+1})$ in a replay memory.  
> - **Step 2**: Training  
>   > - Randomly sample a small batch of experience tuples from replay memory.  
>   > - Use gradient descent to update the Q-network.

> [!summary] Improvements to Deep Q-Learning  
> - **Experience Replay**:  
>   > - Store experiences in a replay memory.  
>   > - Sample uniformly during training to reduce correlations in data.  
> - **Fixed Q-Targets**:  
>   > - Use two networks:  
>   >   - Q-network to select actions.  
>   >   - Target network to evaluate actions.  
>   > - Update the target network less frequently.  
> - **Double Deep Q-Learning**:  
>   > - Mitigate overestimation of Q-values by using separate networks for action selection and evaluation.

---
# Policy Gradient 

> [!note] Policy-based methods
> - Directly learn to approximate $\pi^*$ without having to learn a value function
> - So we **==parameterize the policy==** using a neural network. 
> - $$\boxed{\large \pi_\theta (s) = P(A|s;\theta)} $$
> > [!goal] 
> > Maximize the performance of the parameterized policy using gradient ascent

> [!attention] Pros and Cons of Policy Gradient Methods
> **Pros**
> - Directly estimate policy without storing additional data
> - Can learn a stochastic policy 
> 	- Don't need to implement an exploration/expolitation tradeoff
> - Much more effective in high-dimensional (continuous) action spaces
> - Better convergence properties
> **Cons**
> - They converge to a local max instead of a global max
> - Can take longer to train
> - Can have higher variance


> [!example] Policy Gradient Algorithm  
> 1. **Initialize** the policy parameters $\theta$ randomly.  
>    > - Define the policy $\pi_\theta(s)$ as a parameterized probability distribution over actions.  
> 2. **Generate an episode** by following the current policy $\pi_\theta$:  
>    > - Observe states $s_t$, take actions $a_t$, and receive rewards $r_t$ for $t = 1, 2, \dots, T$.  
> 3. **Calculate the cumulative reward for each step**:  
>    > $$
>    > G_t = \sum_{k=t}^T \gamma^{k-t} r_k,
>    > $$  
>    > where $\gamma$ is the discount factor.  
> 4. **Compute the policy gradient**:  
>    > $$
>    > \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) G_t \right].
>    > $$  
>    > - Estimate this gradient by sampling episodes and using Monte Carlo methods.  
> 5. **Update the policy parameters** using gradient ascent:  
>    > $$
>    > \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta),
>    > $$  
>    > where $\alpha$ is the learning rate.  
> 6. **Repeat steps 2–5** until convergence or for a predefined number of iterations.  

> [!question] Why does the policy gradient work?  
> - The gradient $\nabla_\theta J(\theta)$ pushes the policy to assign higher probabilities to actions that lead to higher rewards.  
> - This allows the policy to improve iteratively by maximizing the expected cumulative reward.  

> [!question] Variants of Policy Gradient  
> - **REINFORCE**: A simple implementation using Monte Carlo estimates for $G_t$.  
> - **Actor-Critic**: Combines a parameterized policy (actor) with a value function (critic) to reduce variance in gradient estimation.  
> - **Proximal Policy Optimization (PPO)**: Regularizes policy updates to ensure stability and prevent large changes in the policy.  

## Policy Gradient Theorem
> [!example] Policy Gradient Theorem  
> 1. **Objective Function**:  
>    > - The goal is to maximize the expected cumulative reward under a parameterized policy $\pi_\theta$:  
>    > $$
>    > J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r_t \right].
>    > $$  
>    > - Here, $\gamma \in [0, 1]$ is the discount factor.  
> 2. **Policy Gradient Theorem**:  
>    > - The gradient of $J(\theta)$ is given by:  
>    > $$
>    > \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a) \right].
>    > $$  
>    > - This connects the expected reward to the gradient of the policy's log-probability, weighted by the action-value function $Q^{\pi_\theta}(s, a)$.  
> 3. **Simplified Gradient Estimate**:  
>    > - Using the cumulative reward $G_t$ as an estimate of $Q^{\pi_\theta}(s, a)$:  
>    > $$
>    > \nabla_\theta J(\theta) \approx \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) G_t \right].
>    > $$  

> [!question] Key Advantages of Policy Gradient Theorem  
> - Directly optimizes the policy without needing a value function.  
> - Works well for stochastic or continuous action spaces.  
> - Facilitates modeling of complex, parameterized policies.  

> [!question] Variants and Improvements  
> - **Baseline Subtraction**:  
>    > - Reduce variance by subtracting a baseline $b(s)$ from $G_t$:  
>    > $$
>    > \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) (G_t - b(s)) \right].
>    > $$  
>    > - Common choice for $b(s)$ is the value function $V^{\pi_\theta}(s)$.  
> - **Actor-Critic Methods**:  
>    > - Use a critic to estimate $Q^{\pi_\theta}(s, a)$ or $G_t$ to reduce variance and stabilize training.  

# Actor-Critic Methods

> [!fail] Variance Problem in REINFORCE  
> - **REINFORCE** relies on Monte Carlo sampling to estimate the gradient:  
>   > $$
>   > \nabla_\theta J(\theta) \approx \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) G_t \right].
>   > $$  
> - The estimate of $G_t$ (cumulative reward) can have **high variance**, especially when:  
>   > - Rewards are sparse or delayed.  
>   > - The episode length is long.  
>   > - The policy $\pi_\theta$ explores random or suboptimal actions frequently.  
> 
> >[!question] Why is high variance a problem?  
> >  - High variance causes instability in training.  
> > - The policy parameters $\theta$ may oscillate or fail to converge.  
> > - Learning slows down as the policy struggles to distinguish between good and bad updates.  
> 
> 
>> [!example] Intuition for the Variance Problem  
> > - Consider an environment where reward depends heavily on the final action in a long episode.  
> > - If actions earlier in the episode are unrelated to the reward, their gradients still get updated based on noisy estimates of $G_t$, leading to inefficient learning.  


> [!summary] Actor-Critic Methods  
> - **Actor**: Parameterizes the policy $\pi_\theta(s) = P(A | s; \theta)$.  
> - **Critic**: Estimates the value function $V(s)$ or action-value function $Q(s, a)$.  
> - **Training**:  
>   > - The actor updates the policy using policy gradients.  
>   > - The critic updates the value estimates using temporal difference (TD) learning.  
>   

> [!example] A2C (Advantage Actor-Critic)  
> 
> - A2C is a reinforcement learning algorithm that combines **actor-critic** methods and uses the **advantage function** to improve training stability and efficiency.  
> - In A2C, the **actor** learns the policy $\pi_\theta(s)$, while the **critic** learns the value function $V^\pi(s)$.  
>   > - The advantage function is defined as:  
>   > $$
>   > A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s),
>   > $$  
>   > which represents how much better the action $a$ is compared to the average action at state $s$.  
>   - The actor is updated using the policy gradient:  
>   > $$
>   > \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) A^\pi(s, a) \right].
>   > $$  
>   - The critic is updated using the temporal difference (TD) error:  
>   > $$
>   > \delta_t = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t).
>   > $$  
>   - **Key Benefits of A2C**:  
>     > - Reduces variance by using the advantage function.  
>     > - More stable than basic REINFORCE, as it incorporates a value function.
>

> [!example] A3C (Asynchronous Advantage Actor-Critic)  
> - A3C extends A2C by using **multiple asynchronous agents** that update the global model in parallel.  
> - These agents each interact with their own environment and compute gradients, updating the global parameters asynchronously.  
> - The **global network** aggregates updates from multiple workers to improve training speed and stability.  
> - **Key Benefits of A3C**:  
>   > - Asynchronous updates prevent correlated gradients and reduce the risk of local minima, leading to faster convergence.  
>   > - It can explore different parts of the environment simultaneously, leading to better generalization.  
>   - **Architecture of A3C**:  
>     > - Each worker runs a separate instance of the environment and computes gradients based on its experiences.  
>     > - Gradients from each worker are asynchronously sent to a global network, which updates the shared parameters.  
>     > - The global network combines the benefits of multiple workers to converge faster than a single worker.  


> [!question] How A3C Improves Over A2C?  
> - **Parallelism**: A3C uses multiple agents (workers) running in parallel, allowing for asynchronous updates to the global model, which improves training efficiency and exploration.  
> - **Reduced Overfitting**: As different workers interact with different environments, A3C reduces the likelihood of overfitting to a single environment.  


---
# Proximal Policy Optimization
> [!summary] Proximal Policy Optimization (PPO)  
> Proximal Policy Optimization (PPO) is an on-policy reinforcement learning algorithm designed to improve the stability and performance of policy gradient methods. PPO addresses key issues like high variance and large policy updates, which can destabilize training. The core idea is to **limit the size of policy updates** to ensure smooth and safe exploration of the policy space.

> [!note] Algorithm
> - The objective function in PPO is designed to **maximize the expected cumulative reward** while ensuring that the new policy does not deviate too far from the old one. The update rule is:
>   $$
>   \max_\theta \mathbb{E}_{\pi_\theta} \left[ \min \left( r_t(\theta) \hat{A}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A} \right) \right].
>   $$  
>   > - $r_t(\theta) = \frac{\pi_\theta(a | s)}{\pi_{\text{old}}(a | s)}$: **Ratio of new to old policy probabilities** at timestep $t$. This ratio quantifies how much the policy has changed.  
>   > - $\hat{A}$: **Advantage estimate**, typically computed using a Generalized Advantage Estimation (GAE) to reduce bias and variance.  
>   > - $\epsilon$: A small hyperparameter that controls the extent to which the policy is allowed to change. Typical values range from 0.1 to 0.3.

> [!question] Why is clipping used in PPO?  
> - **Clipping** is a key component of PPO that ensures stable updates by limiting the likelihood ratio $r_t(\theta)$, preventing overly large updates that could destabilize training.  
> - By clipping $r_t(\theta)$ within the range $[1 - \epsilon, 1 + \epsilon]$, PPO avoids situations where the policy change is too large. This means that when the ratio is outside this range, the objective function will not continue to increase, thus maintaining stability.  
>   > - This clip ensures that updates only happen when the ratio between the old and new policies is within a small, safe range.  

> [!example] PPO Update Rule Explained  
> - The objective function in PPO has two terms:  
>   1. **The original objective**: $r_t(\theta) \hat{A}$ — where we multiply the ratio by the advantage estimate. This encourages actions that lead to higher advantages (i.e., higher returns compared to the average).
>   2. **The clipped objective**: $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}$ — where the ratio is clipped to lie within the range $[1 - \epsilon, 1 + \epsilon]$. If the ratio exceeds this range, the objective function remains constant and does not increase further, preventing large policy updates.
> - The `min` operator ensures that the final objective function is the **smaller** of the two terms, effectively ensuring that the policy updates do not deviate drastically from the previous policy.  

> [!question] Why does PPO work well?  
> - **Efficiency**: PPO strikes a balance between the sample efficiency of methods like TRPO (Trust Region Policy Optimization) and the simplicity of simpler policy gradient methods like REINFORCE.  
> - **Stability**: By clipping the probability ratios, PPO ensures that updates are stable, avoiding issues with overly large policy changes.  
> - **Scalability**: PPO can be applied to large-scale problems with high-dimensional state and action spaces, making it suitable for environments like robotics, games, and simulated environments.

> [!idea] PPO vs. Other Policy Optimization Methods  
> - **TRPO (Trust Region Policy Optimization)**:  
>   > - TRPO uses a constraint on the KL divergence between the old and new policies to ensure safe policy updates. However, TRPO is computationally expensive because it involves solving a constrained optimization problem.  
>   > - **PPO** simplifies this by using clipping rather than a trust region, making it easier to implement and more efficient without sacrificing much performance.  
> - **A3C**:  
>   > - While A3C uses multiple parallel workers to asynchronously update a global network, PPO is a more sample-efficient method that does not require parallelism and can achieve similar performance on many tasks.  
> - **REINFORCE**:  
>   > - PPO outperforms REINFORCE because REINFORCE has higher variance and lacks the stability that comes with PPO's clipping mechanism.  

> [!question] Key Components of PPO  
> 1. **Advantage Estimation**:  
>    > - Typically, a **Generalized Advantage Estimation (GAE)** is used to compute the advantage function, which combines Monte Carlo returns and temporal difference methods to reduce bias and variance.  
> 2. **Policy Updates**:  
>    > - The policy is updated by taking steps based on the gradient of the objective function. The key difference from basic policy gradients is the use of the clipped objective function, which provides stability during updates.  
> 3. **Entropy Regularization (Optional)**:  
>    > - To encourage exploration and prevent premature convergence, PPO often adds an **entropy term** to the objective function. The entropy term penalizes overly deterministic policies, promoting exploration.  
>    > - The entropy regularization is typically added as:  
>    > $$
>    > \mathbb{E}_{\pi_\theta} [\log \pi_\theta(a | s)].
>    > $$  

> [!question] Hyperparameters for PPO  
> - **$\epsilon$ (Clipping Parameter)**:  
>    > - Controls how much the new policy can deviate from the old policy. A typical range is between 0.1 and 0.3.  
> - **Learning Rate**:  
>    > - The step size used in the optimizer, which controls how much the policy is updated at each iteration.  
> - **Batch Size**:  
>    > - The number of experience samples used in each update. PPO often uses **mini-batch optimization** with experience replay.  
> - **Entropy Coefficient**:  
>    > - The weight of the entropy term in the objective function, which controls the exploration-exploitation tradeoff.

---
# Multi-Agent Reinforcement Learning

> [!summary]
> - Extends RL to environments with multiple interacting agents.  
> - **Challenges**:  
>   > - Non-stationarity: Agents' policies change over time.  
>   > - Scalability: Large state-action spaces.  
> - **Approaches**:  
>   > - **Centralized Training, Decentralized Execution**: Train a joint policy but allow decentralized decision-making.  
>   > - **Multi-Agent Policy Gradients**: Extend policy gradients to handle multiple agents.  


# References

[Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)

[CS 285 Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/)

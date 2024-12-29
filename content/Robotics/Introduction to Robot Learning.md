---
date: 2024-12-29
tags:
  - robotics
  - "#notes"
---

> Course Notes for CMU 16-831 Graduate Course on Robot Learning: [Introduction to Robot Learning | Introduction to Robot Learning](https://16-831-s24.github.io/)

# Imitation Learning

>[!info] MDP (Markov Decision process)
>**Definitions**:
  >- $S$: State space, $s_t \in S$: state at time $t$  
  >- $A$: Action space, $a_t \in A$: action at time $t$  
  >- $p$: Transition probability, $s_{t+1} \sim p(\cdot | s_t, a_t)$  
  >- $r$: Reward function, $r: S \times A \to \mathbb{R}$  
  >
> **Goal:** Learn a policy $\pi(a_t | s_t)$.


>[!info] POMDP (Partially Observed MDP)  
>**Additional Definitions**:
  >- $O$: Observation space, $o_t \in O$: observation at time $t$  
  >- $h$: Observation model, $o_t \sim h(\cdot | s_t)$  
  > 
 > **Goal:** Learn a policy $\pi(a_t | o_t)$.


>[!important] Imitation Learning
>>[!note] Idea
>>- collect expert data (observation/state and action pairs)
>>	- Train a function to map observations/states to actions
>>- 
## Dataset Aggregation (DAgger)  
1. **Process**:  
   - Start with expert demonstrations.  
   - Train policy $\pi_1$ via supervised learning.  
   - Run $\pi_1$, query the expert to correct mistakes, and collect new data.  
   - Aggregate new and old data, retrain to create $\pi_2$.  
   - Repeat the process iteratively.  

2. **Advantages**:  
   - Reduces cascading errors.  
   - Provides theoretical regret guarantees.  

3. **Limitations**:  
   - Requires frequent expert queries.  
## IL with Privileged Teachers
- It can be hard to directly learn the policy $\pi_\theta(a_t,o_t)$ especially if $o_t$ is high-dimensional
>[!check] Obtain a "privileged" teacher $\pi_p(a_t, o_t)$
>- $p_t$ contains "ground truth" information that is not available to the "students"
>- Then use $\pi_p(a_t, o_t)$ to generate demonstrations for $\pi_\theta(a_t,o_t)$ 

>[!example] Example 
>- Stage 1: learn a “privileged agent” from expert
> 	- It knows ground truth state (traffic light, other vehicles’ pos/vel, etc)
> - Stage 2: a sensorimotor student learns from this trained privileged agent
> 
> This is especially useful in simulation, because we know  every variable's value in sim. So the privileged teacher learns from that, but the student only learns from stuff it can directly see/measure.
> - privileged teacher is usually trained by PPO

>[!example] Variants
>- Student learning in the latent space: [Adapting Rapid Motor Adaptation for Bipedal Robots](https://ashish-kmr.github.io/a-rma/)
>- Student learning to predict rays: [[2401.17583] Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion](https://arxiv.org/abs/2401.17583)


## Deep Imitation Learning with Generative Modeling

>[!question] What is the problem posed by generative modeling?
>- Learn: learn a distribution $p_\theta$ that *matches* $p_data$
>- Sample: Generate novel data so that $x_{new} \sim p_{\theta}$

For robotics, we want our $p_{data}$ to be from experts. 
There are three leading approaches:
![[Pasted image 20241228201411.png]]

>[!GAIL] GAN + Imitation Learning $\Rightarrow$ Generative Adversarial Imitation Learning (GAIL)
>- Sample trajectory from students
>- Update the discriminator, which is aimed at classifying the teacher and the student
>- Train the student policy which aims to minimize the discriminator's accuracy

>[!note] VAE + IL $\Rightarrow$ Action Chunking with Transformers
>-  Based on CVAE (conditional VAE)
>- Encoder: expert action sequence + observation -> latent
>- Decoder: latent + more observation -> action sequence
prediction
>- Key: action chunking + temporal ensemble


>[!note] Diffusion + IL $\Rightarrow$ Diffusion Policy
>- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)


# Model-Free RL

See [[An Overview of Deep Reinforcement Learning]]. 
# Model-Based RL


# Offline RL



# Bandits and Exploration



# Robot Simulation & Sim2Real


>[!quote] Train in simulation, deploy in real world (with real-time adaptation)

>[!question] Why simulators for robot learning?
>- most RL-based algos are very sample inefficient
>- They are cheap/fast/scalable


>[!fail] Problems of Sim2Real
>- non-parametric mismatches (simulator doesn't consider some effects at all)
>	- complex aerodynamics, fluid dynamics, tire dynamics, etc
>- Parametric mismatches (simulator uses different parameters than real)
>	- robot mass/friction,etc

>[!Success]  Domain Randomization
>- Randomize $e$ in $x_{t+1} = f(x_t, u_t, e)$ 
>- Train a single RL policy $\pi(x)$ that works for many $e$
>	- Approximation of robust control

>[!success] Learning to Adapt
>- Randomize $e$ in $x_{t+1} = f(x_t, u_t, e)$
>- Train an **adaptive** RL policy $\pi(x,e)$ that works for many $e$
>	- approximation of **adaptive control**
>- *Issue!* $e$ is often unknown in **real**
>	- Solution! Learning from a **privileged teacher**
>		- Sim: First Train a teacher policy with privileged information $\pi(x,e)$
>		- Sim: Student policy $\pi_s (x, \text{available info in the real})$ learns from $\pi(x,e)$
>		- Real: Deploy student policy $\pi_s(x, \text{available info in the real})$
>	- Basically an Imitation Learning problem 









# Safe Robot Learning
- 






# Multi-task and Adaptive Robot Learning


# Foundation Models for Robotics

A more comprehensive list: [JeffreyYH/Awesome-Generalist-Robots-via-Foundation-Models: Paper list in the survey paper: Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis](https://github.com/JeffreyYH/Awesome-Generalist-Robots-via-Foundation-Models) 


![[Pasted image 20241228225953.png|]]

- VLMs for Robotic Perception
	- CLIport
	- GeFF
- LLMs for Task Planning
	- SayCan
- LLMs for Action Generation
	- Reward Generation: [Language to Rewards for Robotic Skill Synthesis](https://language-to-reward.github.io/)
- Robotics Foundation Models
	- VLAs
# Future Directions
![[Pasted image 20241228225930.png]]

- Improving Simulations and Sim2Real
- LLM for Reward Design with Eureka
- Doing imitation learning
- No simulator. Collect data from real -> learn a model -> design a policy -> deploy
- Meta-learned dynamics model + online adaptive control
---
layout: post
permalink: /notes/datasci/probability
date: 2024-06-07 15:06:00
title: A Primer on Probability
description: Summary of some key concepts in probability theory.
tags:
  - notes
---


# Bayes Rule

  

> $$ \boxed{P(A|B) = \frac{P(B|A)P(A)}{P(B)}} $$

  
  

Where:

- $$ P(A) $$ = prior;

- $$ P(B \mid A) $$ = likelihood;

- $$ P(A \mid B) $$ = posterior;

  

# Random Variables

  

## Probability Density Function Formula

  

The cumulative distribution function (CDF) is a function that gives the probability that a random variable takes on a value less than or equal to a given value. It is denoted as $$F(x)$$. The CDF is defined as:

  

$$ \boxed{F(x) = P(X \leq x)} $$

  

Where $$ X $$ is the random variable.

  

## Cumulative Distribution Function as an Integral

  

$$ \boxed{F(x) = \int_{-\infty}^{x} \rho(y) dy} $$

  

Where $$ \rho(y) $$ is the probability density function (PDF) of the random variable $$ X $$.

  

# Joint Probability Distributions

  

The joint probability density function (PDF) is a function that describes the probability of multiple random variables taking on specific values simultaneously. It is denoted as $$ f(x_1, x_2, ..., x_n) $$, where $$ x_1, x_2, ..., x_n $$ are the values of the random variables.

  

Properties:

- Non-negativity: $$ f(x_1, x_2) \geq 0 $$ for all $$ x_1, x_2 $$.

- Normalization: $$ \int\int f(x_1, x_2) dx_1 dx_2 = 1 $$, where the integral is taken over the entire range of each random variable.

  

The joint cumulative distribution function (CDF) is a function that gives the probability that multiple random variables take on values less than or equal to given values simultaneously. It is denoted as $$ F(x_1, x_2, ..., x_n) $$, where $$ x_1, x_2, ..., x_n $$ are the values of the random variables.

  

$$ F(x_1, x_2, ..., x_n) = P(X_1 \leq x_1, X_2 \leq x_2, ..., X_n \leq x_n) $$

  

For a joint distribution of multiple random variables $$ X_1, X_2, ..., X_n $$, the joint CDF $$ F(x_1, x_2, ..., x_n) $$ is given by:

  

$$ F(x_1, x_2, ..., x_n) = \int...\int f(x_1, x_2, ..., x_n) dx_1 dx_2 ... dx_n $$

  

This integral formula allows us to calculate the probability that multiple random variables take on values less than or equal to given values simultaneously.

  

# Probability Distributions

  

## Binomial Distribution

  

The binomial distribution gives the probability of $$ k $$ number of successes in $$ n $$ independent trials, where each trial has a probability $$ p $$ of success.

  

$$ \boxed{P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}} $$

  

**Characteristics:**

- **Mean:** $$ np $$

- **Variance:** $$ \sigma^2 = np(1-p) $$

  

**Examples:**

- Coin flips

- Binary outcome events

  

## Poisson Distribution

  

The Poisson distribution gives the probability of the number of events occurring within a fixed interval where the known, constant rate of each event's occurrence is $$ \lambda $$.

  

$$ \boxed{P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}} $$

  

**Characteristics:**

- **Mean:** $$ \lambda $$

- **Variance:** $$ \lambda $$

  

**Examples:**

- Assessing counts over a continuous interval

  - Number of visits to a website in a certain period of time

  

# Continuous Probability Distributions

  

## Uniform Distribution

  

The uniform distribution implies that all outcomes are equally likely within a given range.

  

**PDF:**

$$ \boxed{f(x) = \begin{cases} \frac{1}{b-a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}} $$

  

**Characteristics:**

- **Mean:** $$ \frac{a+b}{2} $$

- **Variance:** $$ \frac{(b-a)^2}{12} $$

  

**Examples:**

- Rolling a fair die

- Sampling (random number generation)

  

## Exponential Distribution

  

The exponential distribution represents the probability of the interval length between events of a Poisson process having a set rate parameter of $$ \lambda $$.

  

**PDF:**

$$ \boxed{f(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x \geq 0 \\ 0 & \text{otherwise} \end{cases}} $$

  

**Characteristics:**

- **Mean:** $$ \frac{1}{\lambda} $$

- **Variance:** $$ \frac{1}{\lambda^2} $$

- **Memoryless Property:** $$ P(X > s+t \mid X > s) = P(X > t) $$

  

## Normal Distribution

  

The normal distribution is characterized by a bell-shaped curve where the mean, median, and mode are all equal.

  

**PDF:**

$$ \boxed{f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}} $$

  

**Characteristics:**

- **Mean:** $$ \mu $$

- **Variance:** $$ \sigma^2 $$

  

# Markov Chains

  

* probability of being in a particular state is only dependent on the previous state

* state space: set of all possible states

* transition matrix: $$ P = [p_{ij}] $$ where $$ p_{ij} $$ is the probability of transitioning from state $$ i $$ to state $$ j $$

  

$$

P = \begin{bmatrix}

p_{11} & p_{12} & \dots & p_{1n} \\

p_{21} & p_{22} & \dots & p_{2n} \\

\vdots & \vdots & \ddots & \vdots \\

p_{n1} & p_{n2} & \dots & p_{nn} \\

\end{bmatrix}

$$

  

Each row of the matrix represents the probabilities of transitioning from a particular state to all other states. The sum of probabilities in each row should be equal to 1.

  
  

* recurrent state: a state that can be reached again from itself

* transient state: a state that cannot be reached again from itself

stationary distribution for markov chain: $$ \pi = \pi P $$

    * $$ P $$ is the transition matrix
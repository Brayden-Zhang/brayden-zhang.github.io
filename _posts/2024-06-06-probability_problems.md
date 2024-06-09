---
layout: post
title: Some Probability Problems
date: 2015-10-20 11:12:00-0400
description: Some easier probability problems and solutions.
tags: math
categories: sample-posts
related_posts: false
---



### Probability of Series Going to 7 Games

**Question:** Two teams play a series of games in a best-of-7 series. The first team to win 4 games wins the series. The teams are evenly matched, so each team has a 50% chance of winning each game. What is the probability that the series goes to 7 games?

<details>
  <summary><strong>Solution</strong></summary>
  
  To find the probability that the series goes to 7 games, we need to consider the scenario where the first 6 games are split 3-3. Since each team has a 50% chance of winning each game, the probability of this scenario occurring is the probability of getting 3 wins out of 6 games for one team and 3 wins out of 6 games for the other team. 

  The numerator $$\binom{6}{3}$$ represents the number of ways to choose 3 wins out of 6 games. 

  The denominator <span style="display: inline;">$$2^6$$</span>  represents the total number of possible outcomes for 6 games.
  
  $$ P(\text{3 wins out of 6}) = \frac{\binom{6}{3}}{2^6}  $$
  
  
  $$ P(\text{3 wins out of 6}) = \frac{20}{64} = 0.3125 $$
  
  Therefore, the probability that the series goes to 7 games is $$\boxed{0.3125}$$ or 31.25%.
</details>




### Bug Reaching a Point in 3D Space

**Question:** 
For a bug that starts at (0,0,0). How many ways are there for the bug to get to (4, 4, 4), if the bug can only move up, right, forward. 

<details>
    <summary><strong>Solution</strong></summary>
    There are 12 moves the bug can make to get to (4, 4, 4). 

    There are therefore $$12!$$ ways to order the 12 total moves. 
    
    The bug must make 4 moves up, 4 moves right, and 4 moves forward. 

    But since each up/right/forward move is indistinguishable, we must divide by $$ 4! $$ for each direction. 

    Therefore, the number of ways the bug can get to (4, 4, 4) is 

    $$ \frac{12!}{4!4!4!} = 34650 $$
</details>
 



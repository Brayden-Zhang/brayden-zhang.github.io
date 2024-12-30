---
tags:
  - notes
  - robotics
date: 2024-12-30
---

# Kinematics and Dynamics

Frames of Reference:
- Global Frame
- Local Frame 

Translation vector ($t$):
$t = [x, y, z]^T$

Rotation Matrix ($R$):

$$
R = \begin{bmatrix}
\cos\theta & \sin\theta & 0 \\
-\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$


> [!note] Homogeneous Coordinates and Transformations
>- **Purpose**: Represent both rotation and translation in a single framework.
>- **Matrix Structure**:$$ T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$$
  >  - $R$: Rotation matrix (3x3).
  >  - $t$: Translation vector (3x1).


Example:
Multiplying $T$ with a point vector (in homogenous coordinates) transforms it into a new frame:
$$\mathbf{p}_{world} = T \cdot \mathbf{p}_{robot}$$​

# Control 

# Motion Planning
> [!example] Finding the shortest path from point A to point B, avoiding any obstacles

> [!important] Dijkstra's Algorithm
> **Description:**
> Dijkstra's algorithm computes the shortest path in a graph with non-negative edge weights. It systematically explores nodes with the smallest accumulated cost.
> **Mathematics:**
> Given a graph $G = (V, E)$, where $V$ is the set of vertices (nodes) and $E$ is the set of edges:
> - $c(u, v)$: Cost of the edge between node $u$ and $v$.
> - $d(v)$: Accumulated cost to reach node $v$.
> 1. Initialize:
>    $d(s) = 0, \quad d(v) = \infty \; \text{for } v \neq s$
>    $s$: Start node.
> 2. Iteratively update:
>    $d(v) = \min(d(v), d(u) + c(u, v)) \; \forall \, (u, v) \in E$
> 3. Stop when all nodes are visited, or the goal node is reached.
> **Steps:**
> - Initialize a priority queue (min-heap) to store nodes with their costs.
> - Set the cost to the start node as 0 and all others as infinity.
> - While the queue is not empty:
>   - Extract the node with the smallest cost.
>   - Update the costs of its neighbors if a shorter path is found.
> - Stop when the goal node is reached.

> **Key Intuition:**
> Imagine you are in a city, and you want to find the least expensive route to a destination. Dijkstra explores all cheaper routes first, ensuring the result is optimal.

> **Advantages:**
> - Guarantees the shortest path.
> - Works well in static environments.

> **Drawbacks:**
> - Computationally expensive for large graphs.
> - Does not scale well for dynamic or real-time scenarios.

> [!important] A* Search
> **Description:**
> A* extends Dijkstra’s algorithm by introducing a heuristic $h(n)$ to guide the search.
> The cost function becomes:
> $f(n) = g(n) + h(n)$
> where:
> - $g(n)$: Cost to reach the current node.
> - $h(n)$: Heuristic cost estimate to reach the goal.

> **Steps:**
> 1. Initialize:
>    $f(s) = g(s) + h(s), \quad g(s) = 0, \quad h(s) = \text{heuristic from } s \text{ to goal}$
>    $s$: Start node.

> 2. While the queue is not empty:
>    - Extract the node with the lowest $f(n)$.
>    - Update the costs for neighbors based on the cost function:
>      $f(v) = g(v) + h(v)$
>    - Continue until the goal node is expanded.

> **Key Intuition:**
> A* balances exploration (using $g(n)$) and goal-directed behavior (using $h(n)$). It’s like finding the shortest path in a maze while guessing how far the exit is.

> **Advantages:**
> - Guarantees the shortest path if $h(n)$ is admissible (i.e., never overestimates the cost).
> - More efficient than Dijkstra for many cases because it guides the search.

> **Drawbacks:**
> - Performance heavily depends on the quality of the heuristic $h(n)$.
> - If $h(n)$ is too weak, A* degenerates to Dijkstra’s algorithm.

> [!question] How do you turn a continuous state/action space into a sparse representation (like a graph)?

> [!important] Rapidly-exploring Random Tree (RRT)
> **Description:**
> RRT is a popular probabilistic algorithm for motion planning in high-dimensional spaces. It incrementally builds a tree by randomly sampling the space and connecting new nodes to the existing tree using a local planner.

> **Steps:**
> 1. Initialize a tree with the start node.
> 2. Sample a random point in the space.
> 3. Find the nearest node in the tree and attempt to extend the tree toward the random point.
> 4. Repeat until a path to the goal is found or a maximum number of iterations is reached.

> **Key Intuition:**
> RRT randomly explores the space, ensuring that even in complex environments with obstacles, the tree quickly spreads and covers much of the space.

> **Advantages:**
> - Can handle high-dimensional spaces.
> - Suitable for nonholonomic and constrained systems.

> **Drawbacks:**
> - The path generated may not be optimal.
> - May require additional steps like path smoothing for practical use.

> [!important] Probabilistic Road Maps (PRMs)
> **Description:**
> PRM is another probabilistic algorithm for motion planning, particularly in complex environments. It constructs a roadmap by randomly sampling the free space and connecting the samples to form a graph.

> **Steps:**
> 1. Sample points randomly in the free space.
> 2. For each pair of nearby samples, check if the path between them is collision-free and add an edge if valid.
> 3. Query the roadmap for a path between the start and goal by connecting the start and goal to the graph and searching for a valid path.

> **Key Intuition:**
> PRMs create a roadmap of valid paths in the environment, making it easier to find paths from start to goal by simply navigating through the pre-built map.

> **Advantages:**
> - Works well in complex and high-dimensional environments.
> - Once the roadmap is constructed, queries for new paths are fast.

> **Drawbacks:**
> - Construction phase can be slow and computationally expensive.
> - Works best in environments where the free space is much larger than the obstacles.

> [!important] Visibility Graph Path Planning
> **Description:**
> Visibility graph planning connects the vertices of obstacles (such as corners) in an environment and creates a graph of visible paths between these vertices.
> **Steps:**
> 1. Identify critical points (e.g., corners of obstacles).
> 2. Connect these points with edges if the line of sight between them is free of obstacles.
> 3. Search the graph for the shortest path.
> 
> **Key Intuition:**
> Visibility graphs simplify the search by only considering important points in the environment, reducing complexity.
> 
> **Advantages:**
> - Provides exact, efficient paths when the environment is relatively simple.
> - Easy to implement in lower-dimensional spaces.
> 
> **Drawbacks:**
> - Computationally expensive in complex or high-dimensional environments.
> - Not suitable for dynamic or rapidly changing environments.

> [!important] Path Smoothing
> **Description:**
> Path smoothing is the process of refining a path generated by a path-planning algorithm to reduce sharp turns or improve efficiency while maintaining feasibility in the environment.

> **Steps:**
> 1. Evaluate the path for long, straight sections or sharp angles.
> 2. Modify the path to smooth out the turns and reduce path length by introducing new intermediate points.

> **Key Intuition:**
> Imagine you have a jagged path through a forest; smoothing it out makes the trajectory smoother and more natural, without veering off course.

> **Advantages:**
> - Improves the quality of the path by making it smoother and less aggressive.
> - Reduces wear and tear on robots and vehicles.

> **Drawbacks:**
> - Smoothing can sometimes lead to paths that are slightly longer.
> - May not be suitable for nonholonomic systems that require sharp turns.




# Linear Quadratic Regulator

>[!caution] Assumptions
> - Dynamics model of the system is known
> - System is linear: $x_{t+1} = Ax_t+Bu_t$ 
> 	- $x_{t+1}$ is the next state of the system
> 	- $u_t$ is the action applied to the system

>[!goal]
>  Stabilize the system around state $x_t= 0$ with control $u_t$ = 0. Then $x_{t+1}=0$ the systems remains at zero forever. 




>[!example] 
>- Helicopter dynamics
>- https://github.com/BracketBotCapstone/quickstart



---
# GraphSLAM

> [!important] GraphSLAM
> **Description:**
> GraphSLAM is an approach for solving the Simultaneous Localization and Mapping (SLAM) problem. It formulates SLAM as a graph-based optimization problem where nodes represent robot poses or landmarks, and edges represent constraints derived from sensor measurements.

> **Key Intuition:**
> - Imagine your path as a connect-the-dots drawing, where each dot is a robot's pose and lines are constraints derived from your sensor data. GraphSLAM optimizes the positions of the dots and lines to make everything fit perfectly.

> **Mathematics:**
> The graph optimization problem is expressed as:
> $$ \text{minimize } \sum_{(i,j) \in E} || z_{ij} - h(x_i, x_j) ||_{\Omega_{ij}}^2 $$
> Where:
> - $x_i$: State of node $i$ (e.g., robot pose or landmark position).
> - $z_{ij}$: Measurement or observation between nodes $i$ and $j$.
> - $h(x_i, x_j)$: Measurement model predicting $z_{ij}$ based on the states $x_i$ and $x_j$.
> - $\Omega_{ij}$: Information matrix (inverse covariance of measurement noise).

> **Steps:**
> 1. Build the graph with nodes for poses and landmarks.
> 2. Add edges based on constraints from odometry and sensor data.
> 3. Use nonlinear optimization techniques (like Gauss-Newton or Levenberg-Marquardt) to minimize the error.

> **Challenges:**
> - **Loop Closure:** Integrating new observations of previously visited areas without introducing inconsistencies.
> - **Computational Cost:** Large graphs with many nodes and edges can become computationally expensive.

> **Intuition for Optimization:**
> - Think of a tangled web of strings representing constraints. Optimization is like pulling the strings until the web is evenly stretched without breaking.

---
# Kalman Filters

> [!important] Kalman Filters
> **Description:**
> Kalman filters are recursive Bayesian filters used for state estimation in systems with linear dynamics and Gaussian noise. They predict the state of a system based on its current estimate and measurements while accounting for uncertainty.

> **Key Intuition:**
> - Imagine walking through a foggy field with a flashlight. Your movement predicts where you’ll be, but the flashlight (sensor) refines your guess based on what it reveals.

> **Mathematics:**

> **Prediction:**
> $$ \hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1} + B u_k $$
> $$ P_{k|k-1} = F P_{k-1|k-1} F^\top + Q $$

> **Update:**
> $$ K_k = P_{k|k-1} H^\top (H P_{k|k-1} H^\top + R)^{-1} $$
> $$ \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1}) $$
> $$ P_{k|k} = (I - K_k H) P_{k|k-1} $$

> Where:
> - $F$: State transition matrix.
> - $Q$: Process noise covariance.
> - $H$: Observation matrix.
> - $R$: Measurement noise covariance.

> **Challenges:**
> - Linear and Gaussian assumptions can be limiting for nonlinear or non-Gaussian systems.
> - Sensitive to parameter tuning and sensor noise.

> **Applications:**
> - Used in tracking (e.g., GPS localization).
> - Integral to sensor fusion systems for robotics.

> **Comparison to EKF:**
> EKF handles nonlinear dynamics by linearizing the system about the current estimate, making it applicable to more complex robotic systems.



# Visual Odometry and Visual SLAM

> [!important] Visual Odometry and Visual SLAM
> **Description:**
> Visual Odometry (VO) refers to the process of estimating a robot's pose (position and orientation) by analyzing consecutive images from a camera. Visual Simultaneous Localization and Mapping (Visual SLAM) extends VO by simultaneously building a map of the environment while estimating the robot's pose within that map.

> **Key Intuition:**
> - Imagine you're walking blindfolded and trying to figure out where you are by feeling the ground with your feet (VO). Now, imagine also sketching a map of the room while you move to keep track of places you've visited (SLAM).
> - VO tracks how the world changes in the camera's view, while SLAM combines this motion with building a spatial representation of the environment.

> [!example] Key Challenges:
> - **Epipolar Constraints:** These ensure that when you look at the same point in two images, the relative camera motion aligns them geometrically. Think of this as connecting two dots on a string.
> - **Depth Estimation:** For stereo cameras, the distance between the cameras acts like human eyes determining depth by how much an object "shifts" between views.
> - **Scale Ambiguity:** A single camera doesn't inherently know how "big" things are in the real world unless you tell it. It's like seeing a photo of a car without knowing if it's a toy or real.
> - **Motion Estimation:** Imagine estimating the robot’s movement from a sequence of slightly blurry pictures—small errors can add up over time, leading to drift.

> **Mathematics:**

> **Epipolar Geometry and Epipolar Constraint:**
> The epipolar constraint helps relate the same 3D point across two different camera views:
> $$ x_2^\top F x_1 = 0 $$
> Where:
> - $x_1$ and $x_2$ are the homogeneous coordinates of a point in the first and second image, respectively.
> - $F$ is the fundamental matrix, encoding the relative camera motion and intrinsic parameters.

> **Depth from Stereo Disparity:**
> Stereo cameras capture the same scene from two viewpoints. The disparity (difference in pixel locations of the same point) is inversely proportional to the depth:
> $$ Z = \frac{f \cdot B}{d} $$
> Where:
> - $Z$ is the depth.
> - $f$ is the focal length of the camera.
> - $B$ is the baseline (distance between the two cameras).
> - $d$ is the disparity (difference in pixel coordinates).

> **Triangulation as a Least-Squares Problem:**
> To estimate the 3D coordinates of a point, triangulation minimizes the reprojection error:
> $$ \text{minimize} \sum_{i=1}^N ||x_i - P_i X||^2 $$
> Where:
> - $X$ is the 3D point in homogeneous coordinates.
> - $P_i$ is the projection matrix for camera $i$.
> - $x_i$ is the 2D observation of the point in image $i$.

> **Scale Issues in Monocular Visual Odometry:**
> Monocular systems only recover relative scale because they lack a direct reference for real-world distances. This is like trying to judge the size of a mountain in a photograph—it depends on your assumptions.

> **Visual SLAM vs. Structure from Motion (SfM):**
> - **Visual SLAM:** Operates online, prioritizing real-time map building and pose estimation.
> - **Structure from Motion (SfM):** Works offline, focusing on building highly detailed 3D models from images, often without real-time constraints.

> **Key Intuition for SLAM:**
> Think of SLAM as solving a jigsaw puzzle where each piece is a local map from sensor data. The robot must place and adjust the pieces while figuring out its own position relative to the puzzle.

# References


[CSC477](https://www.cs.toronto.edu/~florian/courses/csc477_fall22/)
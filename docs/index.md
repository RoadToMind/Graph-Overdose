---
layout: default
title: "ABD-Net: From Reinforcement Learning to Imitation Learning"
description: "An independent implementation and extension of the Articulated-Body Dynamics Network from reinforcement learning to imitation learning."
---

# ABD-Net: From Reinforcement Learning to Imitation Learning
**Independent Implementation and Extension**

Reconstructing a dynamics-grounded robot policy from its paper, evaluating it with PPO, and exploring whether its structural prior can be transferred to imitation learning and manipulation.

**Fawad Hussain**

[Main Paper]({{ site.abd_paper_url }}) · [Implementation]({{ site.source_repository_url }}) · [Project Repository]({{ site.project_repository_url }})

---

# 1. Introduction

GNNs (Graph Neural Networks) use the same deep learning techniques, but apply them to graph-structured data [[1]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-1), [[2]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-2). Using graphs as the main structure, one can still arrive at many popular architectures, such as convolutional and attentional architectures. This is not to say that there are no differences; one of the main properties required is the use of permutation-invariant functions for learning representations of unordered sets [[1]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-1).

Graph neural networks can be used in many areas, such as scene representation, molecules/materials, social networks, and robot kinematics and dynamics, which will be discussed further here.

In robotics, GNNs can be used to build the main structural outline of a robot, including its joints, links, and the type of information that can be transmitted through the graph. A graph consists of nodes and edges: the objects being represented and the connections through which information flows between them.

The deep learning component depends on what exactly we are trying to learn. There are node-level tasks, which predict properties of individual nodes; edge-level tasks, which reason about connections between nodes; and graph-level tasks, which predict properties of the graph as a whole. GNNs therefore commonly follow a graph-in, graph-out architecture, where node, edge, and global embeddings are progressively transformed while preserving the connectivity of the graph [[1]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-1).

This is particularly useful in robotics because a normal MLP largely treats the robot state as one flat vector. A robot, however, is not naturally a flat vector. It is an articulated structure consisting of links, joints, parent-child relationships, forces, and motion propagating throughout the system.

One class of GNNs is the message-passing GNN, where each node looks at its neighbors, combines the information it receives from them, and uses that information to update its own representation. Repeating this process allows information to travel through the entire structure [[1]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-1), [[2]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-2).

## The main paper: ABD-Net

Existing robotic GNN policies already exploit kinematic structures such as link connectivity, providing a framework that can represent the structure of different robots. However, kinematic connectivity alone does not describe how the robot actually behaves dynamically. The propagation of forces and motion through the robot was still relatively underexplored.

This is where the main paper, **ABD-Net** [[3]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-3), comes into play. It asks whether introducing a forward-dynamics-inspired structure into the policy could act as an inductive bias and help the policy learn more effectively.

ABD-Net imposes a meaningful direction of information flow through bottom-up, physics-inspired propagation, together with learnable parameters analogous to inertia-like information and permitted motion directions.

![ABD-Net pipeline]({{ '/assets/images/abdnet-pipeline.svg' | relative_url }})

ABD-Net consists of the following main components:

### 1. Observation Encoding

For each link, the corresponding observation is transformed into an observation embedding.

### 2. Dynamics-Informed Message Passing

Each link first constructs a dynamics-aware representation using its local observation embedding, its learned inertia-like base feature $B$, and the messages received from its descendants.

Before this representation is passed to its parent, the components associated with the learned motion basis $W$ are attenuated. The parent then aggregates the incoming contributions to form its own link representation [[3]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-3).

### 3. Action Decoding

Each joint action is predicted from its parent link representation. The parent representation is useful because it has already incorporated the filtered contribution of the child and its subtree, giving the decoder a more complete representation of the dynamics surrounding that joint [[3]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-3).

### 4. The Added Loss

The orthogonality loss encourages the parameter $W$ for each link to behave like a proper motion basis, helping the approximation used in the message-passing equation remain reasonable [[3]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-3).

### 5. PPO

PPO is the reinforcement-learning algorithm used to learn the policy [[4]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-4), while ABD-Net serves as the architecture embedded inside that policy.

---

# 2. Independent Implementation

Since no official implementation of ABD-Net was available, the method had to be reconstructed directly from the paper.

In my implementation, I tried to stay as close as possible to the described method but had to resolve several ambiguities in the implementation details.

The orthogonality loss was computed separately for each sample and then averaged, rather than first averaging the representations, since these two operations are not equivalent.

The projection term used in the child-to-parent message was bounded to the range [0, 1] to prevent unstable amplification during message propagation, since the orthogonality constraint is only enforced as a soft objective during training.

Small differences in PPO, such as minibatch size, learning rate, value-loss coefficient, and entropy coefficient, can also cause noticeable differences in performance.

There will therefore inevitably be some discrepancies between the implementation described in the paper and this reproduction, but it can still provide insight into how such an implementation behaves, what it is actually learning, and how effective it is.

---

# 3. Part I - Reinforcement Learning Evaluation

For the experiments, I used SAPIEN [[5]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-5), ManiSkill3 [[6]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-6), and its PPO training setup, following the general setup used in the original paper.

The implementation was similarly evaluated on tasks such as the humanoid and hopper environments [[3]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-3).

## Results

## Rollout

## Discussion

A useful inductive bias does not necessarily increase the expressive power of the policy. Instead, it restricts or structures the search space so that the policy is encouraged, in this case, toward more physically meaningful representations.

An interesting aspect, also explored in the original paper, is that the SAPIEN experiments use ManiSkill's default reward functions without the additional gait-related terms used in some of the Genesis locomotion environments [[3]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-3).

Under the limited tasks and seeds I tested, the structural prior did not produce a clear gain in either sample efficiency or final return, while introducing additional computational overhead.

It should also be noted that this implementation did not use the JAX implementation used for some of the experiments in the original work.

---

# 4. Part II - From Reinforcement Learning to Imitation Learning

This raises a broader question: **what happens when the idea is taken beyond the source paper, particularly toward imitation learning (IL), and how far does the usefulness of the dynamics-grounded representation extend?**

This direction connects to previous work that bridges learned expert policies with imitation learning or incorporates graph- and kinematics-based structural priors into imitation-learning policies [[7]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-7)-[[9]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-9).

Tasks such as picking, pulling, and manipulation more generally could potentially benefit from such a physics-inspired prior. Manipulation itself is also mentioned as one of the future directions in the original ABD-Net paper [[3]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-3).

## Transferring ABD-Net to Diffusion Policy

Still using ManiSkill as the backbone, I used its existing Diffusion Policy implementation for the imitation-learning experiments.

Diffusion Policy formulates robot action generation as a conditional denoising diffusion process and has shown strong performance across a range of manipulation tasks [[10]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-10).

Transferring the ABD-Net framework into an IL setting required several components of the original formulation to be replaced.

In my implementation, I removed the original action decoder and PPO component and instead concatenated the learned ABD representation with the pre-existing observation, which then becomes part of the condition supplied to the U-Net in the Diffusion Policy.

One difficulty with this approach is that an embedding exists for every link, and given the size of each embedding, concatenating all of them does not scale particularly well.

I therefore tested several variations, ranging from concatenating only the root-node representation to using representations from all nodes with different embedding sizes.

Among the ABD-based approaches, using the root representation provided the most practical trade-off. Since the propagation is bottom-up, information from the descendants eventually reaches the root. However, compressing everything into only the root representation also means that a considerable amount of link-specific information may be lost.

---

# 5. Imitation-Learning Experiments

For selecting the experiments, two criteria were used:

1. Pre-existing demonstrations had to be available in ManiSkill.
2. The task had to involve either meaningful dynamics or robot configurations where information about the robot's articulated pose could potentially be beneficial.

Based on these criteria, three experiments were chosen:

- **RollBall-v1**
- **PushT**
- **LiftPegUpright**

Three policy variants were compared:

| Policy | Description |
| --- | --- |
| **NormalDiff** | Standard Diffusion Policy using the original observation. |
| **AllComb** | ABD features are concatenated with the original observation. |
| **GraphOnly** | The graph representation provides the main structural conditioning. |

GraphOnly is particularly useful because it forces the policy to rely much more strongly on information derived from the graph, allowing us to examine how plausible and informative the learned graph representation is by itself.

---

# 6. Franka Graph Implementation

Before presenting the results, one important implementation detail concerns the Franka robot used in all three environments.

After modifying the encoder for the imitation-learning setting, where PPO was no longer involved, the original Franka graph did not include the gripper links.

This occurred because the grippers are connected through fixed links, while the graph constructed up to that point only contained dynamic links.

This can be addressed by backtracking each dynamic link to its nearest dynamic parent and connecting the corresponding nodes with an edge.

For **AllComb**, I kept the graph without the additional gripper links because the original observation already contained this information.

For **GraphOnly**, however, these links were incorporated so that the graph itself retained more of the robot's structure.

GraphOnly also requires information about the task object. To provide this, a "dummy" object node was introduced and connected directly to the root node.

This is sufficient for an initial implementation because each node encoder already receives information derived from the overall observation, although the relation between the robot and object is clearly not equivalent to a normal robot joint.

---

# 7. Part II Results

The results show that the best-performing policy overall is **NormalDiff**, followed by **AllComb** and then **GraphOnly**.

Some of the runs also do not appear to have fully converged, which should be considered when interpreting the comparison.

Both AllComb and GraphOnly generally take longer before meaningful performance begins to emerge.

## Evaluation

One possible reason for this underperformance is the limited multimodality of the dataset.

Even in a task with more noticeable dynamics, such as RollBall, the additional structural information may not be necessary if the policy only needs to push the ball and the initial robot configuration remains similar across demonstrations.

In such a setting, detailed information about the robot's full connectivity may provide little additional benefit.

A more informative setting could instead contain a wider range of starting configurations or constrain the available space around the robot so that completing the task requires more difficult and varied poses.

Such changes, however, would also require a new dataset containing demonstrations that cover these configurations.

---

# 8. Conclusion and Future Work

In conclusion, ABD-Net is a constructive push in the right direction, as having a dynamics-informed structure as a prior can be valuable in many tasks.

Although there was not much benefit observed in the experiments conducted here, this is still a relatively small evaluation, and there are several design choices that could be changed to better make use of its potential.

The experiments did not show a consistent performance advantage from transferring the ABD-Net prior directly to manipulation, but further design choices could still be explored, together with better-targeted experiments, to improve approaches such as GraphOnly.

For example, the connection between the object node and the root is currently treated in the same way as the other connections in the graph, even though this is not really the same type of relationship as a joint between two robot links.

There are several graph formulations that could explore this distinction more explicitly.

Another direction would be to avoid adding the object directly to the robot graph and instead use a **scene graph**, where the robot exists as one structured component and the object as another, with a separate relation describing how the two interact.

Recent work has similarly explored scene graphs as explicit structured representations for robotic imitation learning [[11]](https://chatgpt.com/c/6ab3a92a-fbc4-83eb-93d6-d8501df4106d?add_source=github_connector#ref-11).

This could provide a better way of combining the dynamics-informed representation of the robot with the additional relationships required for manipulation.

---

# References

**[1]** B. Sanchez-Lengeling, E. Reif, A. Pearce, and A. B. Wiltschko, "A Gentle Introduction to Graph Neural Networks," *Distill*, 2021, doi: 10.23915/distill.00033.

**[2]** M. M. Bronstein, J. Bruna, T. Cohen, and P. Veličković, "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges," arXiv:2104.13478, 2021.

**[3]** S. Shin, K. Ren, X. Xiong, and J. P. Hanna, "Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning," arXiv:2603.19078, 2026.

**[4]** J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.

**[5]** F. Xiang *et al*., "SAPIEN: A SimulAted Part-Based Interactive ENvironment," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 11097-11107.

**[6]** S. Tao *et al*., "ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI," in *Robotics: Science and Systems (RSS)*, 2025.

**[7]** H. Geng, Z. Li, Y. Geng, J. Chen, H. Dong, and H. Wang, "PartManip: Learning Cross-Category Generalizable Part Manipulation Policy From Point Cloud Observations," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 2978-2988.

**[8]** V. Vosylius and E. Johns, "Instant Policy: In-Context Imitation Learning via Graph Diffusion," in *International Conference on Learning Representations (ICLR)*, 2025.

**[9]** Q. Lv, H. Li, X. Deng, R. Shao, Y. Li, J. Hao, L. Gao, M. Y. Wang, and L. Nie, "Spatial-Temporal Graph Diffusion Policy with Kinematic Modeling for Bimanual Robotic Manipulation," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2025, pp. 17394-17404.

**[10]** C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," in *Robotics: Science and Systems (RSS)*, 2023.

**[11]** J. Qian, Q. Peng, E. Panov, L. Fermoselle, D. Jayaraman, B. Bucher, and T. Kelestemur, "Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs," arXiv:2606.01072, 2026.

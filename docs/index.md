---
layout: default
title: "ABD-NET: From Reinforcement Learning to Imitation Learning"
description: "An independent implementation and extension of the Articulated-Body Dynamics Network."
---

<header class="publication-hero">
<div class="page-container hero-content">
<p class="eyebrow">Independent Implementation and Extension</p>
<h1>
ABD-NET: From Reinforcement Learning
<span class="title-break">to Imitation Learning</span>
</h1>
<p class="hero-summary">
Reconstructing a dynamics-grounded robot policy from its paper,
evaluating it with PPO, and exploring whether its structural prior
transfers to imitation learning and manipulation.
</p>
<p class="authors"><strong>Fawad Hussain</strong></p>
<nav class="publication-links" aria-label="Publication links">
<a class="publication-button" href="{{ site.abd_paper_url }}" target="_blank" rel="noopener noreferrer">Main Paper</a>
<a class="publication-button" href="{{ site.source_repository_url }}" target="_blank" rel="noopener noreferrer">Implementation</a>
<a class="publication-button" href="#part-i-results">Results</a>
<a class="publication-button" href="#references">References</a>
</nav>
</div>
</header>

## Introduction

Modern robot learning increasingly relies on neural networks to learn control policies from interaction data or demonstrations. However, the choice of architecture determines what structure the network is encouraged to exploit. While a standard MLP treats the robot state largely as a flat vector, an articulated robot already has a natural structure defined by its links and joints. Graph Neural Networks (GNNs) provide a natural way of incorporating this structure directly into the policy.

Graph Neural Networks (GNNs) apply deep-learning techniques to graph-structured data <a href="#ref-1">[1]</a>, <a href="#ref-2">[2]</a>. Using graphs as the main structure, one can still arrive at many popular architectures, such as convolutional and attentional architectures.

This structured representation becomes particularly interesting in robotics because the physical structure of a robot is itself naturally graph-like.

This is not to say that there are no differences; one of the main properties required is the use of permutation-invariant functions for learning representations of unordered sets <a href="#ref-1">[1]</a>. Graph neural networks can be used in many areas, such as scene representation, molecules/materials, social networks, and robot kinematics and dynamics, which will be discussed further here.

In robotics, GNNs can be used to build the main structural outline of a robot, including its joints, links, and the type of information that can be transmitted through the graph. A graph consists of nodes and edges: the objects being represented and the connections through which information flows between them.

A graph provides a direct representation of this articulated structure. Robot links can be represented as nodes, while joints form edges connecting them. For example, in a Franka robot, the base and subsequent arm links form a connected chain through the robot's joints. This preserves which parts of the robot are physically connected and allows information to propagate through this structure rather than immediately mixing all joint information into a single flat representation.

The deep learning component depends on what exactly we are trying to learn. There are node-level tasks, which predict properties of individual nodes; edge-level tasks, which reason about connections between nodes; and graph-level tasks, which predict properties of the graph as a whole. GNNs therefore commonly follow a graph-in, graph-out architecture, where node, edge, and global embeddings are progressively transformed while preserving the connectivity of the graph <a href="#ref-1">[1]</a>.

One class of GNNs is the message-passing GNN, where each node looks at its neighbors, combines the information it receives from them, and uses that information to update its own representation. Repeating this process allows information to travel through the entire structure <a href="#ref-1">[1]</a>, <a href="#ref-2">[2]</a>.

The question is therefore not only whether a robot can be represented as a graph, but also what information should propagate through that graph and how. Standard message passing allows connected components to exchange information, while ABD-NET goes further by structuring this information flow according to articulated-body dynamics.

### The Main Paper: ABD-NET

Existing robotic GNN policies already exploit kinematic structures such as link connectivity, providing a framework that can represent the structure of different robots. However, kinematic connectivity alone does not describe how the robot actually behaves dynamically. The propagation of forces and motion through the robot was still relatively underexplored.

Previous robot-learning architectures have already used structural information in different ways. Standard GNN policies use the robot's link and joint connectivity for message passing, while approaches such as BoT and SWAT incorporate robot structure into attention-based architectures. Other approaches incorporate kinematic computation directly into the network. ABD-NET differs by incorporating the computational structure of forward dynamics, using a directed bottom-up information flow inspired by the Articulated Body Algorithm <a href="#ref-3">[3]</a>. The important distinction is therefore not simply that ABD-NET represents the robot as a graph, but that it gives the information passing through that graph a dynamics-inspired structure.

<div class="decision-table-wrapper">
<table class="decision-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Structural information</th>
      <th>Main idea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MLP</td>
      <td>None</td>
      <td>Flat robot-state representation</td>
    </tr>
    <tr>
      <td>Standard GNN</td>
      <td>Link/joint connectivity</td>
      <td>Neighbor message passing</td>
    </tr>
    <tr>
      <td>BoT / SWAT</td>
      <td>Robot structure + attention</td>
      <td>Structure-aware attention</td>
    </tr>
    <tr>
	<td>ABD-NET</td>
      <td>Forward-dynamics structure</td>
      <td>Bottom-up dynamics-informed propagation</td>
    </tr>
  </tbody>
</table>
</div>

This is where the main paper, **ABD-NET** <a href="#ref-3">[3]</a>, comes into play. It asks whether introducing a forward-dynamics-inspired structure into the policy could act as an inductive bias and help the policy learn more effectively.

ABD-NET imposes a meaningful direction of information flow through bottom-up, physics-inspired propagation, together with learnable parameters analogous to inertia-like information and permitted motion directions.

<figure class="wide-figure">
	<img
		src="{{ '/assets/images/abdnet-figure2.png' | relative_url }}"
		alt="Figure 2 from the ABD-NET paper showing observation encoding, dynamics-informed message passing, and action decoding"
	/>
	<figcaption>
		Figure 2 from Shin et al. [3]: Overview of ABD-NET on a quadruped robot.
	</figcaption>
</figure>

ABD-NET consists of the following main components:

<h3>Observation Encoding</h3>
<p>
The input to ABD-NET is the robot observation \(s\). Rather than converting the complete observation directly into one global hidden representation, ABD-NET uses a separate projection \(\phi_i\) for every robot link:
</p>

$$
z_i = \phi_i(s)
$$

<p>
This produces a link-specific embedding \(z_i\). Although each encoder receives information from the robot observation, each projection can learn to extract information that is particularly useful for its corresponding link.
</p>

<h3>From the Robot to the Graph</h3>
<p>
Before message passing can be performed, the robot is represented as a kinematic tree. Each dynamic robot link becomes a node, while joints determine the parent-child connections between nodes. One link is selected as the root, creating a hierarchy through which information can propagate from the outer links toward the root.
</p>

$$
s
\xrightarrow{\Phi}
\{z_i\}
\xrightarrow{\mathcal{M}}
\{v_i\}
\xrightarrow{\Psi}
a
$$

<ul>
<li>\(\Phi\) produces the link-specific observation embeddings.</li>
<li>\(\mathcal{M}\) performs the dynamics-informed message passing.</li>
<li>\(\Psi\) converts the resulting representations into joint actions.</li>
</ul>

### Dynamics-Informed Message Passing

Each link first constructs a dynamics-aware representation using its local observation embedding, its learned inertia-like base feature $$B_i$$, and the messages received from its descendants.

Before this representation is passed to its parent, the components associated with the learned motion basis $$W_j$$ are attenuated. The parent then aggregates the incoming contributions to form its own link representation <a href="#ref-3">[3]</a>.

The bottom-up update is:

$$
v_i=\operatorname{softplus}(z_i+B_i)+\sum_{j\in CH(i)}v_j^a
$$

$$
v_j^a=v_j-v_j\odot(W_jW_j^\top v_j)
$$

Two important learned parameters are introduced during the dynamics-informed message passing.

$$B_i$$ can be thought of as a learned baseline dynamic feature for link \(i\). It plays a role analogous to the rigid-body inertia of a link, although it should not be interpreted as the actual physical inertia.

$$W_i$$ is a learned motion basis. It determines which directions in the latent representation should be attenuated before the contribution of a child link is passed to its parent, analogous to the role of the joint motion subspace in articulated-body dynamics.

ABD-NET therefore does not directly insert the robot's true physical parameters. Instead, it preserves the computational structure of rigid-body dynamics while allowing the corresponding latent quantities to be learned for the control task.

where:

- \(i\): current link
- \(j\): child link of \(i\)
- \(CH(i)\): children of link \(i\)
- \(z_i\): observation embedding
- \(v_i\): dynamics-aware link representation
- \(v_j^a\): filtered contribution passed from child \(j\)
- \(B_i\): learned inertia-like base feature
- \(W_j\): learned motion basis

The softplus operation keeps the representation positive while remaining differentiable, loosely reflecting the positivity properties associated with physical inertia.

### Action Decoding

Each joint action is predicted from its parent link representation. The parent representation is useful because it has already incorporated the filtered contribution of the child and its subtree, giving the decoder a more complete representation of the dynamics surrounding that joint <a href="#ref-3">[3]</a>.

<h3>The Added Loss</h3>
<p>
The orthogonality loss encourages the parameter (W) for each link to behave like
a proper motion basis, helping the approximation used in the message-passing equation
remain reasonable <a href="#ref-3">[3]</a>.
</p>

$$
\mathcal{L}_{\text{orth}}
=
\frac{1}{K}
\sum_i
\left\|
W_i^\top \operatorname{diag}(v_i)W_i-I
\right\|_F^2
$$

<p>
This encourages the learned motion bases to satisfy the approximation used in the dynamics-informed projection <a href="#ref-3">[3]</a>.
</p>

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{PPO}}
+
\lambda_{\text{orth}}\mathcal{L}_{\text{orth}}
$$

<p>
The orthogonality term therefore does not replace the normal PPO objective but instead acts as an additional structural regularizer on the learned ABD representation.
</p>

### PPO

PPO is the reinforcement-learning algorithm used to learn the policy <a href="#ref-4">[4]</a>, while ABD-NET serves as the architecture embedded inside that policy.


<section class="part-banner"><div class="page-container"><h2>From Paper to Implementation</h2></div></section>

## Independent Implementation

Since no official implementation of ABD-NET was available, the method had to be reconstructed directly from the paper. In my implementation, I tried to stay as close as possible to the described method but had to resolve several ambiguities in the implementation details.

The orthogonality loss was computed separately for each sample and then averaged, rather than first averaging the representations, since these two operations are not equivalent.

The projection term used in the child-to-parent message was bounded to the range [0,1] to prevent unstable amplification during message propagation, since the orthogonality constraint is only enforced as a soft objective during training.

Small differences in PPO, such as minibatch size, learning rate, value-loss coefficient, and entropy coefficient, can also cause noticeable differences in performance.

There will therefore inevitably be some discrepancies between the implementation described in the paper and this reproduction, but it can still provide insight into how such an implementation behaves, what it is actually learning, and how effective it is.

## Part I: Reinforcement Learning Evaluation

### Evaluation

For the experiments, I used SAPIEN <a href="#ref-5">[5]</a>, ManiSkill3 <a href="#ref-6">[6]</a>, and its PPO training setup, following the general setup used in the original paper.

The implementation was similarly evaluated on tasks such as the humanoid and hopper environments <a href="#ref-3">[3]</a>.

**Humanoid Walk** controls a highly articulated body with many coupled joints.

**Hopper** has a much simpler morphology, but successful hopping still requires coordination across the body's articulated chain while maintaining balance. It therefore provides a useful contrast to the more complex Humanoid morphology. Both of these tests are also being done in ABD-NET, and thus provide a useful comparison with the original work.

<h3>Humanoid Walk Learning Curves</h3>

<div class="experiment-grid">
	<figure class="wide-figure">
		<img
			src="{{ '/assets/images/figure3-humanoid-walk-paper.svg' | relative_url }}"
			alt="Humanoid Walk learning curves from Figure 3 of the original ABD-NET paper"
		/>
		<figcaption>
			Original paper — Humanoid Walk. Adapted from Shin et al. [3],
			Figure 3, CC BY 4.0.
		</figcaption>
	</figure>

	<figure class="wide-figure">
		<img
			src="{{ '/assets/images/humanoid-walk-reproduction.svg' | relative_url }}"
			alt="Our Humanoid Walk learning curves comparing ABD-NET and MLP"
		/>
		<figcaption>
			Our reproduction — ABD-NET and MLP.
		</figcaption>
	</figure>
</div>

<h3>Hopper Hop Learning Curves</h3>

<div class="experiment-grid">
    <figure class="wide-figure">
		<div class="hopper-paper-crop">
			<img
				src="{{ '/assets/images/hopper-hop-paper-figure3.png' | relative_url }}"
				alt="Hopper Hop learning curves from Figure 3 of the original ABD-NET paper"
			/>
		</div>
	    <figcaption>
		    Original paper — Hopper Hop. Adapted from Shin et al. [3],
		    Figure 3, CC BY 4.0.
	    </figcaption>
    </figure>

    <figure class="wide-figure">
	    <img
		    src="{{ '/assets/images/HopperHopDiagram.svg' | relative_url }}"
		    alt="Our Hopper Hop learning curves from the ABD-NET reproduction"
	    />
	    <figcaption>
		    Our reproduction — ABD-NET Hopper Hop.
	    </figcaption>
    </figure>
</div>

<h3>Humanoid Walk Policy Comparison</h3>

<div class="experiment-grid">
	<figure class="video-figure">
		<video controls muted loop playsinline preload="metadata">
			<source
				src="{{ '/assets/videos/236abd.mp4' | relative_url }}"
				type="video/mp4"
			/>
		</video>
		<figcaption>
			ABD-NET — Seed 2 — Evaluation return: 969.85
		</figcaption>
	</figure>

	<figure class="video-figure">
		<video controls muted loop playsinline preload="metadata">
			<source
				src="{{ '/assets/videos/236.mp4' | relative_url }}"
				type="video/mp4"
			/>
		</video>
		<figcaption>
			MLP — Seed 2 — Evaluation return: 952.00
		</figcaption>
	</figure>
</div>

### What the experiments showed

The reproduction shows that both ABD-NET and the MLP baseline steadily improve throughout training and eventually reach similar returns. In contrast to the separation shown in the original paper, the two curves remain relatively close throughout much of my experiment.

The evaluation rollouts confirm that both policies do in fact learn behaviors capable of obtaining high task returns. However, due to the lack of additional gait-related objectives and ManiSkill's relatively basic reward function, the resulting motion does not resemble a conventional human walk. One noticeable difference between the two videos is that the hip of the ABD-NET robot appears more stable than that of the MLP policy. This may be related to the additional dynamics-informed structural prior, although this observation is qualitative and was not explicitly measured.

In my reproduction, ABD-NET and the MLP baseline follow very similar learning trajectories on Humanoid Walk. Neither model shows a consistent separation throughout training, and both eventually achieve comparable evaluation returns.

Under this implementation and the limited number of seeds tested, the dynamics-informed architectural prior therefore did not result in a clear improvement in either sample efficiency or final return.

### Interpretation

This does not necessarily mean that the inductive bias contains no useful information. Its usefulness may depend strongly on the environment, morphology, reward formulation, and whether solving the task actually requires the additional structural information provided by the graph.

A useful inductive bias does not necessarily increase the expressive power of the policy. Instead, ABD-NET structures the policy toward representations organized according to the computational pattern of articulated-body dynamics.

The somewhat unnatural walking behavior should also be interpreted in the context of the reward function. The policy is optimized to maximize the environment return rather than explicitly produce a human-like gait.

The SAPIEN experiments use ManiSkill's default reward formulation without several of the additional gait-related objectives used in some of the Genesis locomotion environments <a href="#ref-3">[3]</a>. As a result, a mechanically unusual motion can still be considered successful as long as it satisfies the task objective and receives a high return.

It should also be noted that this implementation did not use the JAX implementation used for some of the experiments in the original work.


<section class="part-banner part-banner-secondary"><div class="page-container"><h2>From Reinforcement Learning to Imitation Learning</h2></div></section>

## Part II: From Reinforcement Learning to Imitation Learning

This raises a broader question: **what happens when the idea is taken beyond the source paper, particularly toward imitation learning (IL), and how far does the usefulness of the dynamics-grounded representation extend?**

This direction connects to previous work that bridges learned expert policies with imitation learning or incorporates graph- and kinematics-based structural priors into imitation-learning policies <a href="#ref-7">[7]</a>-<a href="#ref-9">[9]</a>.

Tasks such as picking, pulling, and manipulation more generally could potentially benefit from such a physics-inspired prior. Manipulation itself is also mentioned as one of the future directions in the original ABD-NET paper <a href="#ref-3">[3]</a>.

Consider the same end-effector position being reached using two different Franka arm configurations. Although the gripper may be at nearly the same pose, the shoulder and elbow configurations—and therefore the feasible motion directions and response to an action—can be considerably different. A graph representation preserves the articulated chain that produced this configuration rather than treating the individual joint states as unrelated components of a flat observation.

### Transferring ABD-NET to Diffusion Policy

Still using ManiSkill as the backbone, I used its existing Diffusion Policy implementation for the imitation-learning experiments.

Diffusion Policy formulates robot action generation as a conditional denoising diffusion process and has shown strong performance across a range of manipulation tasks <a href="#ref-10">[10]</a>.

Transferring the ABD-NET framework into an IL setting required several components of the original formulation to be replaced.

<!-- RL-to-IL / ABD-NET + Diffusion Policy architecture figure goes here -->

There are many possible ways of doing this.
In my implementation, I removed the original action decoder and PPO component and instead concatenated the learned ABD representation with the pre-existing observation, which then becomes part of the condition supplied to the U-Net in the Diffusion Policy.

One difficulty with this approach is that an embedding exists for every link, and given the size of each embedding, concatenating all of them does not scale particularly well.

I therefore tested several variations, ranging from concatenating only the root-node representation to using representations from all nodes with different embedding sizes.

Among the ABD-based approaches, using the root representation provided the most practical trade-off. Since the propagation is bottom-up, information from the descendants eventually reaches the root. However, compressing everything into only the root representation also means that a considerable amount of link-specific information may be lost.

<!-- Root-only vs all-node / embedding-size experiment graph goes here -->

<p><strong>The experiments were selected using two criteria:</strong></p>

<ul>
	<li>Pre-existing demonstrations had to be available in ManiSkill.</li>
	<li>The task had to involve either meaningful dynamics or robot configurations where information about the robot's articulated pose could potentially be beneficial.</li>
</ul>

<p>Based on these criteria, three experiments were chosen:</p>

<ol>
	<li>RollBall-v1</li>
	<li>PushT</li>
	<li>LiftPegUpright</li>
</ol>

<p>
These three experiments are compared using the normal Diffusion Policy
(<strong>NormalDiff</strong>), the policy where ABD features are concatenated
with the original observation (<strong>AllComb</strong>), and the policy where
the graph representation provides the main structural conditioning
(<strong>GraphOnly</strong>).
</p>

<p>
For example:
</p>

<p>
<strong>NormalDiff:</strong> uses the original state observation supplied by the ManiSkill Diffusion Policy environment.
</p>

<p>
<strong>AllComb:</strong> preserves that complete observation and augments it with the selected ABD embedding.
</p>

<p>
<strong>GraphOnly:</strong> removes/reduces the direct dependence on the flat robot observation and instead conditions the policy primarily on the graph-derived representation; task-object information must therefore also be represented in the graph.
</p>

<p>
GraphOnly is particularly useful because it forces the policy to rely much more
strongly on information derived from the graph, allowing us to examine how
plausible and informative the learned graph representation is by itself.
</p>

<h2 class="implementation-heading">Franka Graph Implementation</h2>

Before presenting the results, one important implementation detail concerns the Franka robot used in all three environments.

After modifying the encoder for the imitation-learning setting, where PPO was no longer involved, the original Franka graph did not include the gripper links. This occurred because the grippers are connected through fixed links, while the graph constructed up to that point only contained dynamic links.

This can be addressed by backtracking each dynamic link to its nearest dynamic parent and connecting the corresponding nodes with an edge.

For **AllComb**, I kept the graph without the additional gripper links because the original observation already contained this information.

For **GraphOnly**, however, these links were incorporated so that the graph itself retained more of the robot's structure.

GraphOnly also requires information about the task object. To provide this, a "dummy" object node was introduced and connected directly to the root node.

This is sufficient for an initial implementation because each node encoder already receives information derived from the overall observation, although the relation between the robot and object is clearly not equivalent to a normal robot joint.


<a id="part-i-results"></a>

## Results

<div class="results-graph-grid">
  <figure>
    <img src="{{ '/assets/images/rollball-results.svg' | relative_url }}"
         alt="RollBall evaluation results" />
    <figcaption>RollBall</figcaption>
  </figure>

  <figure>
    <img src="{{ '/assets/images/pusht-results.svg' | relative_url }}"
         alt="PushT evaluation results" />
    <figcaption>PushT</figcaption>
  </figure>

  <figure>
    <img src="{{ '/assets/images/liftpeg-results.svg' | relative_url }}"
         alt="LiftPegUpright evaluation results" />
    <figcaption>LiftPegUpright</figcaption>
  </figure>
</div>

### What the plots show

**RollBall:** NormalDiff and AllComb begin improving earlier than GraphOnly and maintain stronger performance through most of training. GraphOnly improves considerably more slowly.

**PushT:** All three methods achieve comparatively low and noisy success rates. NormalDiff reaches the strongest performance toward the end of training, while AllComb improves later and GraphOnly remains lower.

**LiftPegUpright:** NormalDiff begins learning substantially earlier. AllComb initially lags behind but improves later in training, while GraphOnly again shows slower and weaker learning.

Taken together, the experiments show a consistent trend: the original Diffusion Policy performs best overall, adding the ABD representation does not provide a clear advantage, and relying primarily on the graph representation makes learning considerably more difficult.

### Evaluation

The results show that the best-performing policy overall is NormalDiff, followed by AllComb and then GraphOnly. Some of the runs also do not appear to have fully converged, which should be considered when interpreting the comparison.

Both AllComb and GraphOnly generally take longer before meaningful performance begins to emerge.

One possible reason for this underperformance is the limited multimodality of the dataset. Even in a task with more noticeable dynamics, such as RollBall, the additional structural information may not be necessary if the policy only needs to push the ball and the initial robot configuration remains similar across demonstrations.

In such a setting, detailed information about the robot's full connectivity may provide little additional benefit.

A more informative setting could instead contain a wider range of starting configurations or constrain the available space around the robot so that completing the task requires more difficult and varied poses.

Such changes, however, would also require a new dataset containing demonstrations that cover these configurations.

## Conclusion and Future Work

In conclusion, ABD-NET is a constructive push in the right direction, as having a dynamics-informed structure as a prior can be valuable in many tasks.

Although there was not much benefit observed in the experiments conducted here, this is still a relatively small evaluation, and there are several design choices that could be changed to better make use of its potential.

The experiments did not show a consistent performance advantage from transferring the ABD-NET prior directly to manipulation, but further design choices could still be explored, together with better-targeted experiments, to improve approaches such as GraphOnly.

For example, the connection between the object node and the root is currently treated in the same way as the other connections in the graph, even though this is not really the same type of relationship as a joint between two robot links.

There are several graph formulations that could explore this distinction more explicitly like Relational Graph Convolutional Network or even give different priorities to different edges using Graph Attention Network

Another direction would be to avoid adding the object directly to the robot graph and instead use a scene graph, where the robot exists as one structured component and the object as another, with a separate relation describing how the two interact.

Recent work has similarly explored scene graphs as explicit structured representations for robotic imitation learning <a href="#ref-11">[11]</a>.

This could provide a better way of combining the dynamics-informed representation of the robot with the additional relationships required for manipulation.


<section id="references" class="paper-section references-section">
<div class="page-container prose">
<h2>References</h2>

<ol class="reference-list">
<li id="ref-1">B. Sanchez-Lengeling, E. Reif, A. Pearce, and A. B. Wiltschko, <em>A Gentle Introduction to Graph Neural Networks</em>, Distill, 2021, doi: 10.23915/distill.00033. <a href="https://distill.pub/2021/gnn-intro/" target="_blank" rel="noopener noreferrer">Paper</a></li>
<li id="ref-2">M. M. Bronstein, J. Bruna, T. Cohen, and P. Veličković, <em>Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges</em>, arXiv:2104.13478, 2021. <a href="https://arxiv.org/abs/2104.13478" target="_blank" rel="noopener noreferrer">Paper</a></li>
<li id="ref-3">S. Shin, K. Ren, X. Xiong, and J. P. Hanna, <em>Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning</em>, arXiv:2603.19078, 2026. <a href="{{ site.abd_paper_url }}" target="_blank" rel="noopener noreferrer">Paper</a></li>
<li id="ref-4">J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, <em>Proximal Policy Optimization Algorithms</em>, arXiv:1707.06347, 2017. <a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noopener noreferrer">Paper</a></li>
<li id="ref-5">F. Xiang <em>et al.</em>, <em>SAPIEN: A SimulAted Part-Based Interactive ENvironment</em>, in <em>Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)</em>, 2020, pp. 11097–11107.</li>
<li id="ref-6">S. Tao <em>et al.</em>, <em>ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI</em>, in <em>Robotics: Science and Systems (RSS)</em>, 2025.</li>
<li id="ref-7">H. Geng, Z. Li, Y. Geng, J. Chen, H. Dong, and H. Wang, <em>PartManip: Learning Cross-Category Generalizable Part Manipulation Policy From Point Cloud Observations</em>, in <em>Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)</em>, 2023, pp. 2978–2988.</li>
<li id="ref-8">V. Vosylius and E. Johns, <em>Instant Policy: In-Context Imitation Learning via Graph Diffusion</em>, in <em>International Conference on Learning Representations (ICLR)</em>, 2025.</li>
<li id="ref-9">Q. Lv, H. Li, X. Deng, R. Shao, Y. Li, J. Hao, L. Gao, M. Y. Wang, and L. Nie, <em>Spatial-Temporal Graph Diffusion Policy with Kinematic Modeling for Bimanual Robotic Manipulation</em>, in <em>Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)</em>, 2025, pp. 17394–17404.</li>
<li id="ref-10">C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, <em>Diffusion Policy: Visuomotor Policy Learning via Action Diffusion</em>, in <em>Robotics: Science and Systems (RSS)</em>, 2023. <a href="https://roboticsproceedings.org/rss19/p026.html" target="_blank" rel="noopener noreferrer">Paper</a></li>
<li id="ref-11">J. Qian, Q. Peng, E. Panov, L. Fermoselle, D. Jayaraman, B. Bucher, and T. Kelestemur, <em>Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs</em>, arXiv:2606.01072, 2026.</li>
</ol>
</div>
</section>

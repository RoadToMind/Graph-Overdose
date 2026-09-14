---
layout: default
title: "ABD-Net: From Paper to Imitation Learning"
description: >-
  A dynamics-grounded graph representation for robot policies, extended from
  reinforcement learning to diffusion-policy imitation learning.
body_class: home-page
---

<section class="hero" id="top">
  <div class="page-shell">
    <div class="hero-copy">
      <p class="eyebrow">Paper → Code → Robot Learning</p>

      <h1>
        Teaching robot policies
        <span class="gradient-text">how bodies are connected.</span>
      </h1>

      <p class="lead">
        I reproduced the Articulated-Body Dynamics Network in PyTorch and DGL,
        integrated it into reinforcement learning, and then reused its
        dynamics-grounded representation for diffusion-policy imitation learning.
      </p>

      <div class="hero-actions">
        <a class="button" href="#journey">
          Explore the project
          <span aria-hidden="true">↓</span>
        </a>

        <a class="button button-secondary" href="{{ '/presentation/' | relative_url }}">
          Read the presentation script
        </a>

        <a
          class="button button-secondary"
          href="{{ site.source_repository_url }}"
          target="_blank"
          rel="noopener noreferrer"
        >
          View the code
          <span aria-hidden="true">↗</span>
        </a>
      </div>
    </div>

    <div class="metric-grid" aria-label="Project statistics">
      <div class="metric"><strong>813</strong><span>Demonstrations</span></div>
      <div class="metric"><strong>37,851</strong><span>Transitions</span></div>
      <div class="metric"><strong>100k</strong><span>Training iterations</span></div>
      <div class="metric"><strong>60%</strong><span>Best root-ABD success</span></div>
    </div>
  </div>
</section>

<section class="section section-border" id="scope">
  <div class="page-shell">
    <div class="notice">
      <strong>Scope:</strong>
      This is an independent implementation and extension of ABD-Net.
      The paper studies ABD-Net as a reinforcement-learning actor. The
      diffusion-policy integration presented here is an extension to imitation
      learning. Local RollBall results currently use one seed and should be
      interpreted as engineering evidence rather than a conclusive benchmark.
    </div>
  </div>
</section>

<section class="section" id="journey">
  <div class="page-shell">
    <div class="section-heading">
      <p class="eyebrow">The journey</p>
      <h2>From a research idea to a tested extension</h2>
      <p class="lead">The project progressed through four stages, with each stage answering a different engineering or research question.</p>
    </div>
    <div class="card-grid card-grid-four">
      <article class="card"><span class="card-number">01 / PAPER</span><h3>Understand the prior</h3><p>Translate the Articulated Body Algorithm's child-to-parent computation into a learnable neural architecture.</p></article>
      <article class="card"><span class="card-number">02 / BUILD</span><h3>Implement ABD-Net</h3><p>Construct kinematic graphs from SAPIEN and implement link encoders, message passing, action heads, and regularization.</p></article>
      <article class="card"><span class="card-number">03 / VALIDATE</span><h3>Integrate with PPO</h3><p>Test the graph actor end to end in ManiSkill locomotion environments with batched training, checkpoints, metrics, and videos.</p></article>
      <article class="card"><span class="card-number">04 / EXTEND</span><h3>Condition diffusion policy</h3><p>Reuse the ABD representation as structural context for imitation learning and compare it with the official diffusion baseline.</p></article>
    </div>
  </div>
</section>

<section class="section section-border" id="architecture">
  <div class="page-shell">
    <div class="section-heading">
      <p class="eyebrow">The paper</p>
      <h2>Connectivity tells us where. Dynamics suggests how.</h2>
      <p class="lead">Standard graph policies use the robot's structure to determine which components communicate. ABD-Net also gives that communication a forward-dynamics-inspired direction and transformation.</p>
    </div>
    <div class="architecture" aria-label="ABD-Net architecture">
      <article class="architecture-node"><span>Φ · ENCODE</span><h3>Link-wise features</h3><p>Every robot link independently maps the complete observation into a link-specific embedding.</p></article>
      <div class="architecture-arrow" aria-hidden="true">→</div>
      <article class="architecture-node"><span>M · PROPAGATE</span><h3>Dynamics messages</h3><p>Features accumulate from child links toward their parents following the kinematic tree.</p></article>
      <div class="architecture-arrow" aria-hidden="true">→</div>
      <article class="architecture-node"><span>Ψ · DECODE</span><h3>Joint actions</h3><p>Each joint action is decoded from the propagated representation of its parent link.</p></article>
    </div>
    <div class="result-callout"><strong>ABA</strong><p>The architecture does not attempt to recreate an exact simulator. Instead, it turns the computational structure of the Articulated Body Algorithm into a learnable inductive bias.</p></div>
  </div>
</section>

<section class="section section-border" id="build">
  <div class="page-shell">
    <div class="section-heading"><p class="eyebrow">Implementation</p><h2>Turning equations into a trainable policy</h2></div>
    <div class="card-grid">
      <article class="card"><span class="card-number">KINEMATIC GRAPH</span><h3>SAPIEN to DGL</h3><p>Controlled joints define the included links. Directed edges point from each child toward its parent, enabling a leaf-to-root topological pass.</p><p>Motor indices are recorded separately so graph order never gets confused with the action order required by the environment.</p></article>
      <article class="card"><span class="card-number">MESSAGE PASSING</span><h3>Stable propagation</h3><p>Each link receives child contributions and combines them with a positive local representation.</p><p>A smooth bounded projection prevents quadratic feature amplification through deeper trees while remaining differentiable.</p></article>
      <article class="card"><span class="card-number">TRAINING</span><h3>End-to-end gradients</h3><p>The orthogonality loss is optimized with the policy objective. Tests confirm that gradients reach both the per-link encoders and dynamics-informed message passing.</p></article>
    </div>
    <div class="hero-actions">
      <a class="button button-secondary" href="{{ site.source_repository_url }}/blob/main/abdnet_actor.py" target="_blank" rel="noopener noreferrer">ABD-Net implementation</a>
      <a class="button button-secondary" href="{{ site.source_repository_url }}/blob/main/ppo.py" target="_blank" rel="noopener noreferrer">PPO integration</a>
      <a class="button button-secondary" href="{{ site.source_repository_url }}/blob/main/tests/test_diffusion_policy_abd.py" target="_blank" rel="noopener noreferrer">Integration tests</a>
    </div>
  </div>
</section>

<section class="section section-border" id="imitation">
  <div class="page-shell">
    <div class="section-heading"><p class="eyebrow">The extension</p><h2>Using ABD-Net for imitation learning</h2><p class="lead">Diffusion policy does not need ABD's action decoder. It uses the link-wise encoder and dynamics propagation as a structural condition for action denoising.</p></div>
    <div class="pipeline" aria-label="Diffusion policy pipeline">
      <div class="pipeline-step"><strong>State history</strong><span>2 observations</span></div><div class="pipeline-step"><strong>Φ encoders</strong><span>Per-link features</span></div><div class="pipeline-step"><strong>M propagation</strong><span>Leaf to root</span></div><div class="pipeline-step"><strong>Condition</strong><span>Raw state + ABD</span></div><div class="pipeline-step"><strong>1D U-Net</strong><span>Predict noise</span></div><div class="pipeline-step"><strong>Action chunk</strong><span>Execute policy</span></div>
    </div>
    <div class="split" style="margin-top: 2rem;">
      <article class="card variant-card"><span class="variant-label">Compact condition</span><h3>Root-conditioned ABD</h3><p>The propagated root representation summarizes information that has travelled through the complete Panda arm tree.</p><p>The root feature is concatenated with the unchanged raw observation before conditioning the diffusion model.</p></article>
      <article class="card variant-card"><span class="variant-label">Detailed condition</span><h3>All-node ABD</h3><p>All eight propagated Panda arm representations are retained, allowing the diffusion policy to access local information from every link.</p><p>Direct and projected representations test whether additional structural detail helps or overwhelms the condition.</p></article>
    </div>
  </div>
</section>

<section class="section section-border" id="results">
  <div class="page-shell">
    <div class="section-heading"><p class="eyebrow">Evaluation</p><h2>Competitive, but not universally better</h2><p class="lead">All main RollBall runs use the same 813 demonstrations, 100,000 iterations, batch size 1,024, seed 1, and 100 evaluation episodes every 5,000 iterations.</p></div>
    <div class="table-shell"><table><thead><tr><th>Policy</th><th>Action horizon</th><th>Best success once</th><th>Best iteration</th><th>Final success once</th><th>Best success at end</th></tr></thead><tbody>
      <tr><td>Official diffusion</td><td>8</td><td>47%</td><td>75k</td><td>39%</td><td>1%</td></tr><tr><td>ABD root</td><td>8</td><td class="result-positive">47%</td><td>75k</td><td>33%</td><td>1%</td></tr><tr><td>Official diffusion</td><td>12</td><td class="result-positive">64%</td><td>70k</td><td>44%</td><td>1%</td></tr><tr><td>ABD root</td><td>12</td><td>60%</td><td>95k</td><td class="result-positive">59%</td><td>1%</td></tr><tr><td>Official diffusion</td><td>15</td><td>55%</td><td>100k</td><td class="result-positive">55%</td><td>2%</td></tr><tr><td>ABD root</td><td>15</td><td class="result-positive">59%</td><td>90k</td><td>54%</td><td>1%</td></tr>
    </tbody></table></div>
    <div class="card-grid" style="margin-top: 2rem;"><article class="card"><span class="card-number">RESULT 01</span><h3>ABD remains competitive</h3><p>Root-conditioned ABD ties the baseline at action horizon 8 and reaches a four-point higher peak at horizon 15.</p></article><article class="card"><span class="card-number">RESULT 02</span><h3>The outcome depends on horizon</h3><p>At horizon 12, the official baseline has the higher peak, although the ABD policy finishes closer to its own peak.</p></article><article class="card"><span class="card-number">RESULT 03</span><h3>More features are not always better</h3><p>Directly supplying all 2,048 ABD features performs worse than a compact or projected all-node condition.</p></article></div>
    <div class="result-callout"><strong>!</strong><p>Success-at-end remains low for every method. The agent can reach the goal transiently, but reliably maintaining task completion is still an open problem.</p></div>
  </div>
</section>

<section class="section section-border" id="next"><div class="page-shell"><div class="section-heading"><p class="eyebrow">Next steps</p><h2>What would turn this into stronger evidence?</h2></div><div class="card-grid"><article class="card"><span class="card-number">STATISTICS</span><h3>Matched multi-seed runs</h3><p>Repeat the official, root, and projected all-node variants with at least three seeds and report confidence intervals.</p></article><article class="card"><span class="card-number">GENERALIZATION</span><h3>Dynamics shifts</h3><p>Change mass, payload, friction, and damping to test the central robustness claim behind the dynamics-grounded prior.</p></article><article class="card"><span class="card-number">TASK COVERAGE</span><h3>Contact-rich manipulation</h3><p>Extend evaluation to PushT and LiftPegUpright to determine whether the representation generalizes beyond RollBall.</p></article></div></div></section>

<section class="section"><div class="page-shell"><div class="cta"><p class="eyebrow">Present the project</p><h2>Use the complete seven-minute showcase script</h2><p>The script follows the page from the paper's central question through implementation, imitation learning, evaluation, and future work.</p><div class="hero-actions" style="justify-content: center;"><a class="button" href="{{ '/presentation/' | relative_url }}">Open presentation script <span aria-hidden="true">→</span></a><a class="button button-secondary" href="{{ site.paper_url }}" target="_blank" rel="noopener noreferrer">Read the paper <span aria-hidden="true">↗</span></a></div></div></div></section>

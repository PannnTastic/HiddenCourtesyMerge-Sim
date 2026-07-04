# HiddenCourtesyMerge-Sim: Project and POMDP Overview

## What This Project Does

HiddenCourtesyMerge-Sim is a controlled simulation benchmark for studying
whether an ego vehicle can benefit from estimating a hidden driver-courtesy
type during highway merge negotiation.

The project uses `highway-env` `merge-v0` as the base simulator. That simulator
already provides the highway merge scene, vehicles, lanes, and traffic dynamics.
This project adds a controlled hidden variable on top of it:

- `cooperative`: the merge vehicle is parameterized to create or accept safer
  gaps.
- `non-cooperative`: the merge vehicle is parameterized to maintain higher
  speed and smaller headways.

The ego vehicle does not observe this courtesy label directly. It only observes
motion cues such as relative distance, merge-vehicle speed, and acceleration.
The hidden label is saved only for evaluation and oracle baselines.

The central question is:

> Does accurate online inference of hidden driver courtesy improve closed-loop
> merge safety?

The key finding is:

> High offline courtesy-inference accuracy does not necessarily improve
> closed-loop safety, because the belief is least reliable during the early
> interaction window when the ego vehicle must make safety-critical yielding
> decisions.

An analogy: the ego vehicle is like a driver trying to decide whether another
driver will yield. After watching long enough, the driver may infer the other
driver's intent correctly. But the safety-critical decision often has to be
made before enough evidence has accumulated.

## What Is New Relative to `highway-env merge-v0`

The novelty is not the existence of a merge simulator. `highway-env merge-v0`
already provides that.

This project adds:

1. A controlled hidden courtesy variable injected through IDM/MOBIL parameters.
2. Ground-truth hidden labels for evaluation, while policies remain blind to
   the labels.
3. Matched-episode counterfactual evaluation: all policies face the same
   episode seeds, hidden courtesy labels, traffic conditions, and observation
   noise.
4. Bayesian belief tracking over hidden courtesy.
5. Diagnostic comparisons among random, rule, belief, oracle-heuristic, and
   POMCP policies.
6. Failure-mode analysis showing that offline belief accuracy can fail to
   translate into closed-loop safety improvement.

## POMDP Model

The policy-facing problem is modeled as an abstracted hidden-parameter POMDP:

```text
M = (S, A, O, T, Z, R, gamma, b0)
```

This is not a complete enumeration of the full `highway-env` simulator state.
It is an abstracted interface exposing only the variables consumed by the
benchmark policies. The remaining traffic state, road geometry, and simulator
internals remain inside the simulator transition kernel.

## State Space

There are two useful ways to talk about state in this project.

### 1. Full Simulator State

The full simulator state is the internal `highway-env` state:

```text
x_t^env = all vehicle positions, speeds, lanes, road geometry,
          controller parameters, and simulator bookkeeping
```

This full state is what the simulator uses to execute dynamics. It includes
surrounding traffic and lane geometry that are not explicitly exposed in the
compact POMDP tuple.

### 2. Compact POMDP State

The compact policy-facing state is:

```text
s_t = (x_e, v_e, x_m, v_m, d_t, Delta v_t, m)
```

where:

| Symbol | Meaning |
|---|---|
| `x_e` | ego longitudinal position |
| `v_e` | ego speed |
| `x_m` | merge vehicle longitudinal position |
| `v_m` | merge vehicle speed |
| `d_t = x_m - x_e` | signed longitudinal gap |
| `Delta v_t = v_m - v_e` | relative speed |
| `m` | hidden courtesy type |

The state space can be written as:

```text
S = X_e x V_e x X_m x V_m x D x DeltaV x M
```

where:

```text
M = {cooperative, non_cooperative}
```

and `X_e`, `V_e`, `X_m`, `V_m`, `D`, and `DeltaV` are continuous ranges induced
by the simulator. The benchmark policies do not enumerate this state space
explicitly. They consume the current observed features and maintain a belief
only over `M`.

The hidden variable is:

```text
m in {cooperative, non_cooperative}
```

The courtesy type is sampled once at episode reset and remains fixed:

```text
P(m_{t+1} = m_t) = 1
```

This is a deliberate benchmark design choice. The project isolates
hidden-parameter inference from the harder problem of time-varying human intent.

In short:

```text
Full simulator state: used by highway-env to run the world.
Compact POMDP state: used by the paper to formalize hidden courtesy.
Belief state: only over the hidden courtesy variable m.
```

## Actions

The full `highway-env` action space includes meta-actions such as lane changes,
acceleration, maintaining speed, and slowing down. For controlled comparisons,
the structured non-random policies use:

```text
A_structured = {IDLE, SLOWER}
```

This restriction is intentional. It keeps the rule, belief, and
oracle-heuristic policies comparable so the experiment isolates the effect of
hidden courtesy information rather than action-space differences.

The random policy samples from the broader action set and is used only as a
lower-bound sanity check.

### Why Not Use All Native `highway-env` Actions?

The project still uses native `highway-env` meta-actions. It does not introduce
new low-level controls. The structured benchmark policies intentionally use a
subset of the native action space.

The native action set includes:

```text
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
```

The structured policies use:

```text
IDLE, SLOWER
```

This restriction is an experimental-control choice. The main question is not
whether a richer action space can solve merge-v0. The main question is:

> Does hidden courtesy information improve the ego vehicle's yielding decision?

If all actions are enabled, policy differences become harder to interpret. A
policy may perform better because it has better courtesy inference, but it may
also perform better because it can change lane, accelerate, or exploit a more
expressive action space.

By using the same restricted action table for the rule, belief, and
oracle-heuristic policies, the comparison isolates information access:

```text
rule_policy   : same action table, uniform courtesy prior
belief_policy : same action table, Bayesian courtesy belief
oracle_policy : same action table, true courtesy label
```

Thus, differences among these policies can be interpreted as differences in
hidden-courtesy information, not differences in available maneuvers.

The restriction is also a limitation. The benchmark does not claim that
`IDLE/SLOWER` is sufficient for all merge scenarios, and it does not test
whether lateral evasive maneuvers or acceleration policies would dominate in
merge environments where those actions are practically available.

Paper-safe phrasing:

> We use highway-env's native discrete meta-actions, but restrict the structured
> non-random policies to `{IDLE, SLOWER}` to isolate the effect of hidden
> courtesy information from action-space expressiveness.

## Observations

The ego does not observe `m`. It observes motion cues during the interaction
window:

```text
2 m < d_t < 40 m
```

The observation vector is:

```text
o_t = (ard_t, mvs_t, mva_t)
```

where:

| Feature | Meaning |
|---|---|
| `ard` | absolute relative distance |
| `mvs` | merge vehicle speed |
| `mva` | merge vehicle acceleration |

These features are used because they carry information about whether the merge
vehicle is behaving cooperatively or non-cooperatively.

## Observation Model

The canonical observation likelihood is a diagonal Gaussian model:

```text
Z(o_t | m) = product_i N(o_{t,i}; mu_{m,i}, sigma_{m,i}^2)
```

for:

```text
i in {ard, mvs, mva}
```

This model is intentionally simple and reproducible. A full-covariance Gaussian
improves offline belief accuracy, but the main benchmark keeps the diagonal
model as the released baseline.

## Belief Update

The policy maintains a belief over courtesy:

```text
b_t(m) = P(m | o_{1:t})
```

Because courtesy is fixed during an episode, the courtesy transition is:

```text
P(m_{t+1} = m' | m_t = m) =
  1 if m' = m
  0 otherwise
```

The general POMDP belief update would be:

```text
b_{t+1}(m') = eta * Z(o_{t+1} | m') * sum_m P(m' | m) b_t(m)
```

Since `m` is static, this simplifies to:

```text
tilde_b_{t+1}(m) = eta * Z(o_{t+1} | m) * b_t(m)
```

where `eta` normalizes the belief.

To avoid overconfidence from temporally correlated observations, the posterior
is shrunk toward the uniform prior:

```text
b_{t+1} = (1 - lambda) * tilde_b_{t+1} + lambda * b0
```

with:

```text
b0 = [0.5, 0.5]
lambda = 0.08
```

For two courtesy modes, the update can be read intuitively as:

```text
posterior_coop =
  likelihood(observation | cooperative) * prior_coop

posterior_noncoop =
  likelihood(observation | non_cooperative) * prior_noncoop

normalize both so they sum to 1
```

If the merge vehicle slows down in a way that is more likely under the
cooperative model, `b(cooperative)` increases. If it maintains speed or
accelerates in a way that is more likely under the non-cooperative model,
`b(non_cooperative)` increases.

Outside the interaction window, no informative courtesy observation is used, so
the belief is carried forward unchanged except for the shrinkage term.

## Transition Model

The transition model has two parts:

1. the static transition of the hidden courtesy type,
2. the simulator transition of the observable traffic state.

### Hidden Courtesy Transition

The hidden courtesy type is static:

```text
P(m' | m) = 1 if m' = m, else 0
```

This means a cooperative driver does not become non-cooperative halfway through
the episode, and a non-cooperative driver does not become cooperative halfway
through the episode. This is a benchmark simplification, not a claim about real
human psychology.

### Observable Traffic Transition

The observable traffic state is advanced by `highway-env`:

```text
s'obs ~ T_env(s_obs, a, m, xi)
```

where:

| Symbol | Meaning |
|---|---|
| `s_obs` | observable part of the compact state |
| `a` | ego action |
| `m` | hidden courtesy type, implemented through IDM/MOBIL parameters |
| `xi` | simulator randomness and unlisted traffic variables |

Combined with the static hidden state, the abstract transition is:

```text
T(s' | s, a) = delta(m' = m) * T_env(s'_obs | s_obs, a, m)
```

In practice, `T_env` also depends on simulator variables not listed in the
compact state, such as surrounding traffic, lane geometry, and
vehicle-controller internals. Those variables are absorbed into the simulator
kernel.

For the ego vehicle, the action affects the next state through the simulator:

```text
IDLE   -> ego approximately maintains its current behavior
SLOWER -> ego reduces speed, increasing the chance of creating a gap
```

For the merge vehicle, the hidden courtesy type affects motion through its
IDM/MOBIL parameters:

```text
cooperative     -> lower target speed / larger time gap / more gap creation
non_cooperative -> higher target speed / smaller headway / less yielding
```

This is why the state transition depends on `m` even though `m` is not observed
by the ego policy.

## Reward

Evaluation uses the highway-env reward plus a close-call penalty:

```text
R_t = r_env_t - 0.25 * 1[TTC_t < 3 s or d_t in (0, 15 m)]
```

The reward balances speed/throughput with collision and near-collision risk.
The weight is manually chosen and treated as an evaluation parameter, not a
principled safety cost. Robustness checks replay stored rewards under alternate
weights.

## Policies Compared

The project evaluates:

| Policy | What it knows / does |
|---|---|
| `random_policy` | lower-bound sanity check |
| `rule_policy` | uses a fixed uniform courtesy prior |
| `belief_policy` | uses Bayesian belief over courtesy |
| `oracle_policy` | receives the true courtesy label, but only inside the same heuristic action table |
| `pomcp_policy` | online POMCP planning over the hidden courtesy belief |

The oracle is called an oracle-heuristic because it is not an optimal POMDP
oracle. It only has perfect courtesy knowledge inside a restricted heuristic
policy class.

## What the Experiments Show

The main result is not simply that POMCP performs well. The more important
diagnostic result is the separation between inference quality and closed-loop
safety:

1. The Bayesian belief model can classify courtesy accurately offline.
2. The belief-conditioned heuristic does not significantly outperform the
   uniform-prior rule policy in closed-loop evaluation.
3. The failure is concentrated in the early interaction window.
4. The early interaction window is exactly when safety-critical yielding
   decisions must be made.
5. POMCP outperforms the oracle-heuristic because planning can represent
   action sequences that the heuristic action table cannot.

## Scientific Contribution

The project should be framed as a diagnostic benchmark, not as the first POMDP
merge planner.

The core contribution is:

> HiddenCourtesyMerge-Sim provides a controlled matched-episode benchmark for
> testing whether hidden-courtesy inference improves closed-loop merge safety,
> and reveals a failure mode where accurate offline belief estimation arrives
> too late to improve safety-critical closed-loop decisions.

This framing is stronger and safer than claiming novelty from cooperative vs.
non-cooperative merge behavior alone, because prior work already uses hidden
intent and cooperation levels in autonomous-driving POMDPs.

## What This Project Does Not Claim

The project does not claim:

- real-world autonomous-driving performance,
- empirical human-driver psychology,
- first use of POMDPs for merging,
- first use of cooperative/non-cooperative driver types,
- an optimal POMDP solution,
- that POMCP exceeds an optimal oracle policy.

The correct scope is:

> Results characterize this controlled highway-env merge-v0 benchmark under the
> evaluated traffic densities, courtesy parameter ranges, observation model,
> and policy classes.

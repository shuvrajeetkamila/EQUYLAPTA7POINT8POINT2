# MATHEMATICAL SPECIFICATION OF THE DIFFERENTIABLE FUNCTIONAL OBJECTIVE

## EQUYLAPTA 7.8.1: True Functional-Effect Loss Optimization

---

## 1. Motivation: Causal Reconstruction vs. Output Imitation

A fundamental defect in earlier iterations of cross-architecture transfer is the conflation of:
1. **Output Imitation (Behavioral Cloning)**: Forcing the recipient model to mimic the donor's final output logits or token probabilities:
   $$L_{\text{imitation}}(\theta) = -\sum_{v} y_{\text{donor}}(x)_v \log P_{\text{recip}}(x; \theta)_v$$
   This merely trains the recipient to behave like the donor, regardless of whether the transplanted component is causally responsible for the behavior.
2. **Causal Functional-Effect Reconstruction**: Optimizing the recipient adapter parameters such that the *causal change* induced by the transplanted component in the recipient matches the *causal change* induced by the component in the donor:
   $$L_{\text{functional}}(\theta) = D\left( \Delta_{\text{recip}}(x, \theta), \, \Delta_{\text{target}}(x) \right)$$

In EQUYLAPTA 7.8.1, we strictly optimize the **Causal Functional-Effect Reconstruction Objective**.

---

## 2. Mathematical Formulation

Let:
- $x$ be an input sequence of token IDs of length $T$.
- $V$ be the vocabulary size ($V = 2000$).
- $Y_{\text{donor, intact}}(x) \in \mathbb{R}^V$ be the final logit vector of the donor specialist `e4-math-4L` with all candidate circuit elements active.
- $Y_{\text{donor, ablated}}(x) \in \mathbb{R}^V$ be the final logit vector of the donor specialist when the candidate functional circuit is ablated (e.g. `L0_head_2` or the full FDC set to zero).

The **Donor Causal Target Signature** is defined as:
$$\Delta_{\text{target}}(x) = Y_{\text{donor, intact}}(x) - Y_{\text{donor, ablated}}(x) \in \mathbb{R}^V$$

Now let:
- $\theta = \{W_{\text{in}}, W_{\text{out}}\}$ denote the trainable adapter parameters bridging the frozen donor component into the recipient architecture `e6-base-4L`.
- $Y_{\text{recip, intact}}(x, \theta) \in \mathbb{R}^V$ be the recipient model's final logit vector when the translated component is active.
- $Y_{\text{recip, ablated}}(x, \theta) \in \mathbb{R}^V$ be the recipient model's final logit vector when the translated component is ablated (slot zeroed out).

The **Recipient Causal Delta** is defined as:
$$\Delta_{\text{recip}}(x, \theta) = Y_{\text{recip, intact}}(x, \theta) - Y_{\text{recip, ablated}}(x, \theta) \in \mathbb{R}^V$$

The **Primary Differentiable Functional Objective** is formulated as the mean-squared logit discrepancy combined with Frobenius regularization on the adapters:
$$L_{\text{functional}}(\theta) = \frac{1}{2V} \sum_{v=1}^V \left( \Delta_{\text{recip}}(x, \theta)_v - \Delta_{\text{target}}(x)_v \right)^2 + \frac{\lambda_{\text{reg}}}{2} \sum_{W \in \theta} \|W - W_0\|_F^2$$

where:
- $V = 2000$ (normalized over vocabulary dimension).
- $\lambda_{\text{reg}} = 10^{-4}$ prevents drift away from the identity/zero initialization.
- $W_0$ is the initial adapter weight matrix.

---

## 3. Derivation of the Two-Branch Backpropagation Gradient

Let $e(x, \theta) \in \mathbb{R}^V$ denote the functional-effect error residual at the output logits:
$$e(x, \theta) = \Delta_{\text{recip}}(x, \theta) - \Delta_{\text{target}}(x) = (Y_{\text{recip, intact}}(x, \theta) - Y_{\text{recip, ablated}}(x, \theta)) - \Delta_{\text{target}}(x)$$

By the multivariable chain rule:
$$\frac{\partial L_{\text{functional}}}{\partial Y_{\text{recip, intact}}} = \frac{1}{V} e(x, \theta)$$
$$\frac{\partial L_{\text{functional}}}{\partial Y_{\text{recip, ablated}}} = -\frac{1}{V} e(x, \theta)$$

Using the recipient model's native backward differentiation operator `backward(ids, dlogits)`:
1. **Intact Branch**:
   Set `head_mask` to intact state:
   $$G_{\text{intact}} = \text{recipient.backward}\left(x, \, \frac{1}{V} e(x, \theta)\right)$$
2. **Ablated Branch**:
   Set `head_mask` or slot mask to ablated state:
   $$G_{\text{ablated}} = \text{recipient.backward}\left(x, \, -\frac{1}{V} e(x, \theta)\right)$$

The total gradient with respect to the recipient's effective slot weight $W_{\text{eff}}$ is:
$$G_{\text{eff}} = \frac{\partial L_{\text{functional}}}{\partial W_{\text{eff}}} = G_{\text{intact}}[W_{\text{eff}}] + G_{\text{ablated}}[W_{\text{eff}}]$$

Since the effective weight is parameterized via the bidirectional bridge:
$$W_{\text{eff}} = W_{\text{in}} W_{\text{comp}} W_{\text{out}}$$
where $W_{\text{comp}}$ is the **strictly frozen** donor component, the analytical gradients with respect to the trainable adapter matrices are:
$$\frac{\partial L_{\text{functional}}}{\partial W_{\text{in}}} = G_{\text{eff}} (W_{\text{comp}} W_{\text{out}})^T + \lambda_{\text{reg}} (W_{\text{in}} - W_{\text{in}, 0})$$
$$\frac{\partial L_{\text{functional}}}{\partial W_{\text{out}}} = (W_{\text{in}} W_{\text{comp}})^T G_{\text{eff}} + \lambda_{\text{reg}} (W_{\text{out}} - W_{\text{out}, 0})$$

---

## 4. Why This Eliminates Optimization Artifacts

1. **Zero Proxy Gradients**: Every parameter update is derived analytically from the true loss via the exact model computational graph.
2. **Pure Functional Targeting**: The recipient cannot satisfy this objective simply by becoming generally more capable or copying static output labels. It can only satisfy this objective if **ablating the transplanted component causes the exact same logit shift that ablating the component caused in the donor specialist**.
3. **Strict Invariant Guarantee**: $W_{\text{comp}}$ receives zero gradient ($\nabla_{W_{\text{comp}}} = 0$ enforced by construction), preserving strict frozenness throughout optimization.

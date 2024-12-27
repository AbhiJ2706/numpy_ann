This document provides an explanation of the matrix calculations for batch gradient descent in a neural network with 
- $n$ layers
- any number of neurons per layer
- any choice of differentiable loss function

**Definitions**

   **Inputs and Layers:**
   - Batch size: $n$
   - Input features: $d_0$ (number of features in the input layer)
   - $L$: Total number of layers (including the output layer)
   - $d_l$: Number of neurons in layer $l$, for $l = 1, 2, ..., L$.

   **Weights and Biases:**
   - $\tilde{W}_l$: Weight matrix for layer $l$, including biases.

   **Activations and Pre-activations:**
   - $Z_l$: Pre-activation (weighted sum) at layer $l$.
   - $H_l$: Activation at layer $l$ (a#er applying activation function).
   - $\tilde{H}_l$: Activation at layer $l$, augmented with bias column of ones.

   **Loss Function:**
   - $\mathcal{L}$: Loss function applied to the final layer $L$ and ground truth $Y$.

**Forward Propagation**

   **Initialization:**
   - Augment the input matrix with a bias column:
     $$
     \tilde{H}_0 = [X, 1], \quad \tilde{H}_0 \in \mathbb{R}^{n \times (d_0 + 1)}.
     $$

   **Layer-wise Computation:**
   For each layer $l = 1, 2, ..., L$:
   1. Compute pre-activation:
      $$
      Z_l = \tilde{H}_{l-1} \tilde{W}_l, \quad Z_l \in \mathbb{R}^{n \times d_l}.
      $$
   2. Apply activation function $f_l$:
      $$
      H_l = f_l(Z_l), \quad H_l \in \mathbb{R}^{n \times d_l}.
      $$
   3. Augment $H_l$ with a bias column for subsequent layers (if $l < L$):
      $$
      \tilde{H}_l = [H_l, 1], \quad \tilde{H}_l \in \mathbb{R}^{n \times (d_l + 1)}.
      $$

   For the final layer $L$, the output is:
   $$
   H_L = f_L(Z_L), \quad H_L \in \mathbb{R}^{n \times d_L}.
   $$

**Compute Loss**

   The loss function $\mathcal{L}$ depends on the task and the outputs $H_L$:
   - **Classification:** Cross-entropy loss, e.g., binary or categorical.
   - **Regression:** Mean squared error or other loss functions.

   Let:
   $$
   \mathcal{L} = \frac{1}{n} \sum_{i=1}^n \ell(H_L[i], Y[i]),
   $$
   where $\ell$ is the loss for a single sample.
   
**Backward Propagation**

   **Initialization at the Output Layer:**
   1. Compute gradient of the loss w.r.t. the output layer activation:
      $$
      \frac{\partial \mathcal{L}}{\partial H_L}, \quad \text{Dimensions: } \mathbb{R}^{n \times d_L}.
      $$
   2. Backpropagate through the activation function of the output layer:
      $$
      \frac{\partial \mathcal{L}}{\partial Z_L} = \frac{\partial \mathcal{L}}{\partial H_L} \odot f_L'(Z_L), \quad \text{Dimensions: } \mathbb{R}^{n \times d_L}.
      $$

   **Layer-wise Backpropagation:**
   For each layer $l = L, L-1, ..., 1$:
   1. Gradient w.r.t. weights:
      $$
      \frac{\partial \mathcal{L}}{\partial \tilde{W}_l} = \frac{1}{n} \tilde{H}_{l-1}^\top \frac{\partial \mathcal{L}}{\partial Z_l}, \quad \text{Dimensions: } \mathbb{R}^{(d_{l-1} + 1) \times d_l}.
      $$
   2. Backpropagate to the previous layer’s activations (if $l > 1$):
      $$
      \frac{\partial \mathcal{L}}{\partial \tilde{H}_{l-1}} = \frac{\partial \mathcal{L}}{\partial Z_l} \tilde{W}_l^\top, \quad \text{Dimensions: } \mathbb{R}^{n \times (d_{l-1} + 1)}.
      $$
   3. Drop the gradient of the bias term from $\frac{\partial \mathcal{L}}{\partial \tilde{H}_{l-1}}$:
      $$
      \frac{\partial \mathcal{L}}{\partial H_{l-1}} = \frac{\partial \mathcal{L}}{\partial \tilde{H}_{l-1}}[:, :-1], \quad \text{Dimensions: } \mathbb{R}^{n \times d_{l-1}}.
      $$
   4. Backpropagate through the activation function:
      $$
      \frac{\partial \mathcal{L}}{\partial Z_{l-1}} = \frac{\partial \mathcal{L}}{\partial H_{l-1}} \odot f_{l-1}'(Z_{l-1}), \quad \text{Dimensions: } \mathbb{R}^{n \times d_{l-1}}.
      $$

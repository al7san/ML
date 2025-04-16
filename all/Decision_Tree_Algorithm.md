## 🌳 Decision Tree Algorithm

### 📖 Definition
A **Decision Tree** is a tree‑structured model used for classification or regression. Each internal node tests a feature, each branch represents a test outcome, and each leaf node gives a class label (or numeric value).

---

### ✅ Advantages
- **Interpretability**  
  Easy to visualize and explain to non‑technical stakeholders.
- **No Feature Scaling Needed**  
  Works with raw data—no normalization or standardization required.
- **Handles Mixed Data Types**  
  Can process both numerical and categorical features.
- **Implicit Feature Selection**  
  Automatically chooses the most informative features via splits.
- **Fast Inference**  
  Once built, making predictions is very quick (just follow branches).

---

### ⚠️ Disadvantages
- **Overfitting Prone**  
  Deep trees can memorize noise unless pruned or regularized.
- **Unstable**  
  Small changes in data can lead to a completely different tree.
- **Bias Toward Features with More Levels**  
  Splitting criteria may favor variables with many distinct values.
- **Greedy Splitting**  
  Standard algorithms (e.g., CART) optimize one split at a time and may miss globally optimal trees.
- **Limited Predictive Power**  
  Often outperformed by ensemble methods (Random Forests, Gradient Boosting).

---

🎯 **Tip**: Use pruning, set maximum depth, or combine multiple trees (ensembles) to mitigate overfitting and instability.  
---
## 🌳 How to Use a Decision Tree for Classification

A Decision Tree classifies data in two main phases: **training** (building the tree) and **prediction** (traversing it).

---

### 1. Training Phase (Tree Construction)

1. **Prepare the Data**  
   - Collect labeled samples: each record has feature values and a class label.  
   - Handle missing values, encode categoricals if needed.

2. **Choose a Split Criterion**  
   - Common measures: **Gini impurity** or **Information Gain (Entropy)**.  
   - At each node, evaluate every feature (and possible threshold for numeric features) to find the split that best separates the classes.

3. **Split the Data**  
   - Partition the dataset into two child nodes based on the chosen feature test (e.g., `feature ≤ threshold`).  
   - Assign each sample to the left or right child.

4. **Repeat Recursively**  
   - For each child node, repeat steps 2–3 on its subset of data.  
   - Stop when you reach a **leaf condition**, e.g.:  
     - All samples in the node share the same class.  
     - Node depth reaches a maximum.  
     - Node has too few samples (controlled by `min_samples_split`).

5. **Assign Leaf Labels**  
   - Each leaf node predicts the **majority class** of the training samples it contains.

---

### 2. Prediction Phase (Classifying New Samples)

1. **Start at the Root Node**  
   - For a new sample, evaluate the root’s feature test.

2. **Follow the Branch**  
   - If the test is true (e.g., `feature ≤ threshold`), go to the left child; otherwise, go to the right child.

3. **Continue Until a Leaf**  
   - Repeat step 2 at each internal node until you reach a leaf.

4. **Output the Class**  
   - The leaf’s assigned class label is the model’s prediction for that sample.

---

### 3. Quick Example

Given a toy “Iris” dataset with features `PetalLength` and `PetalWidth`:

- **Root split**: `PetalLength ≤ 2.45` → separates **Setosa** vs. others.  
- Next splits on `PetalWidth` to distinguish **Versicolor** vs. **Virginica**.  
- Leaves end up labeled with one of the three iris species.

---

### 4. Tips for Better Classification

- **Control Overfitting**: limit `max_depth`, increase `min_samples_split`, or prune after training.  
- **Handle Imbalance**: adjust class weights or use stratified splits.  
- **Feature Importance**: inspect split frequencies to understand which features matter most.  
---
## 🔢 Calculating Entropy in Decision Trees

Entropy measures the **impurity** or **uncertainty** of a node. A pure node (all samples from one class) has entropy $H(S)=0$, while a perfectly mixed node has maximum entropy.

---

### 📐 Entropy Formula

For a node $S$ with $c$ classes, let $p_i$ be the proportion of samples in class $i$. Then:

$$
H(S)\;=\;-\sum_{i=1}^{c} p_i \,\log_{2}\!p_i
$$

Here:

$$
p_i = \frac{\\text{ samples of class }i}{\text{total \ samples in node}}
$$


We use $\log_{2}$ so entropy is measured in bits.


---

### 🔍 Step‑by‑Step Calculation

1. **Count samples per class** in the node.  
2. **Compute proportions** $p_i$.  
3. **Apply the formula**: sum $-\,p_i\,\log_{2}p_i$ across all classes.  
4. **Interpret**:  
   - $H(S)=0$ ⇒ node is pure (single class).  
   - $H(S)$ is highest when all $p_i$ are equal.

---

### 📊 Example (Binary Classification)

Node has:  
- 9 positives (class +)  
- 5 negatives (class –)  
- Total = 14 samples

Compute proportions:  
$$p_{+} = \frac{9}{14} \approx 0.643,\quad
p_{-} = \frac{5}{14} \approx 0.357
$$

Entropy:  

$$
\begin{aligned}
H(S)
&= -\bigl(p_{+}\log_{2}p_{+} + p_{-}\log_{2}p_{-}\bigr)\\
&= -\bigl(0.643\log_{2}0.643 + 0.357\log_{2}0.357\bigr)\\
&\approx 0.940\text{ bits}
\end{aligned}
$$

---

🎯 **Key Point**:  
When choosing a split, a Decision Tree maximizes **Information Gain**, i.e. the reduction in entropy:

$$
text{Information Gain} = H(\text{parent}) - \sum_{\text{children}} \frac{|\text{child}|}{|\text{parent}|}\,H(\text{child})
$$

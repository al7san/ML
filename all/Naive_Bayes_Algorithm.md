## 🤖 Naive Bayes Algorithm

### 📖 Definition
The Naive Bayes classifier is a probabilistic model based on **Bayes’ theorem**, which for features \(x_1, \dots, x_n\) and class \(y\) states:

$$
P(y \mid x_1,\dots,x_n)
= \frac{P(y)\,\prod_{i=1}^{n}P(x_i \mid y)}{P(x_1,\dots,x_n)},
$$

under the **conditional independence** assumption: each feature \(x_i\) is independent of every other feature given the class \(y\).

---

### ✅ Advantages
- **Efficiency**: Training and prediction are very fast (linear in the number of features and samples).  
- **Low Data Requirement**: Works well even with small datasets.  
- **Scales to High Dimensions**: Handles thousands of features without a significant performance hit.  
- **Robust to Irrelevant Features**: Irrelevant features have minimal impact due to the independence assumption.  
- **Probabilistic Output**: Produces well‑calibrated class probabilities, useful for ranking or thresholding.

---

### ⚠️ Disadvantages
- **Strong Independence Assumption**: Real‑world features are often correlated, which can degrade accuracy.  
- **Zero‑Frequency Problem**: If a feature value never appears in the training data for a class, the likelihood becomes zero (mitigated by techniques like Laplace smoothing).  
- **Probability Estimates May Be Poor**: Violations of model assumptions can lead to unreliable probability estimates.  
- **Limited Expressiveness**: Cannot model feature interactions or complex decision boundaries.

---

## 🤖 How the Naive Bayes Classifier Works & Its Core Assumptions

### 🔄 Working Principle

1. **Bayes’ Theorem**  
   For a sample with features \$(\mathbf{x} = [x_1, \dots, x_n]\)$ and a class \(y\), we compute the posterior probability:
   $$P(y \mid \mathbf{x})
   = \frac{P(y)\,\prod_{i=1}^{n} P(x_i \mid y)}{P(\mathbf{x})}
   \;\propto\; P(y)\,\prod_{i=1}^{n} P(x_i \mid y)
   $$
2. **Estimate Priors**  
   - \$(P(y)\)$ is estimated from training data as the fraction of samples in class \(y\).

3. **Estimate Likelihoods**  
   - For each feature \$(x_i\)$:
     - **Categorical**: \$(P(x_i \mid y)\)$ is estimated by relative frequency.
     - **Continuous** (Gaussian NB): assume \$(x_i \mid y \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)\)$, so
     - /
       $$P(x_i \mid y)
       = \frac{1}{\sqrt{2\pi\sigma_{iy}^2}}
         \exp\!\biggl(-\tfrac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2}\biggr).
       $$

4. **Compute Posteriors & Predict**  
   - For each class \$(y\)$, compute the (log) posterior \$(\log P(y) + \sum_i \log P(x_i \mid y)\)$.  
   - **Predict** the class with the highest posterior:
     $$hat y = \arg\max_{y}\;P(y)\,\prod_{i=1}^n P(x_i \mid y).
     $$

---

### 🧩 Core Assumptions

1. **Conditional Independence**  
   - Given the class \(y\), features \(x_i\) are assumed independent:
     $$P(x_i, x_j \mid y)
     = P(x_i \mid y)\;P(x_j \mid y)
     \quad\forall i \neq j.
     $$

2. **Correct Likelihood Model**  
   - You must choose an appropriate distribution for \(P(x_i \mid y)\) (e.g., multinomial for counts, Gaussian for continuous).

3. **Feature Relevance**  
   - Irrelevant features won’t harm much, but highly correlated features can violate independence and degrade performance.

---

🎯 **Bottom Line**:  
Naive Bayes turns simple frequency or distribution estimates into a fast, probabilistic classifier by assuming that features don’t interact given the class.  


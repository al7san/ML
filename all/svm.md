## 🛡️ SVM (Support Vector Machine) Algorithm

### 📖 Definition
A Support Vector Machine (SVM) finds the hyperplane in feature space that **maximizes the margin** between two classes. Given weights $\mathbf{w}$ and bias $b$, it solves:

$$
\min_{\mathbf{w}, b} \;\frac{1}{2}\|\mathbf{w}\|^2
\quad\text{s.t.}\quad
y_i(\mathbf{w}^\top x_i + b) \ge 1,\;\forall i
$$

---

### ✅ Advantages
- **Effective in high dimensions**: Works well when feature space is large.  
- **Kernel trick**: Can handle non‑linear boundaries via kernels (RBF, polynomial, etc.).  
- **Memory efficient**: Uses only support vectors, not all training points.  
- **Good generalization**: Margin maximization helps reduce overfitting.

---

### ⚠️ Disadvantages
- **Scalability**: Training can be slow for large $N$ (complexity up to $O(N^2)$–$O(N^3)$).  
- **Hyperparameter sensitivity**: Performance depends heavily on $C$ (regularization) and kernel parameters (e.g., $\gamma$).  
- **No direct probability outputs**: Requires additional calibration (e.g., Platt scaling) for probabilities.  
- **Less robust to noisy/overlapping classes**: Hard margins can be overly sensitive to outliers.

---

🎯 **Tip**: For very large datasets, consider a **linear SVM** (e.g., using stochastic solvers) or approximate methods like **SGDClassifier** with hinge loss.  

---
## 🔄 How the SVM Algorithm Works & Support Vectors

### 1. Linear SVM Training
- **Objective**: Find the hyperplane  
  $$f(x)=\mathbf{w}^\top x + b = 0$$  
  that maximally separates two classes.
- **Hard‑Margin Formulation** (linearly separable data):  
  $$min_{\mathbf{w},b}\;\frac{1}{2}\|\mathbf{w}\|^2
  \quad\text{s.t.}\quad
  y_i\bigl(\mathbf{w}^\top x_i + b\bigr)\ge1,\;\forall i.
  $$
- **Soft‑Margin** (allows some misclassifications):  
  $$min_{\mathbf{w},b,\xi}\;\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i}\xi_i
  \quad\text{s.t.}\quad
  y_i(\mathbf{w}^\top x_i + b)\ge1-\xi_i,\;\xi_i\ge0.
  $$

---

### 2. Kernel Trick (Non‑linear SVM)
- Map inputs via \$(\phi(x)\)$ into a higher‑dimensional space.
- Solve the **dual problem** using kernel \$(K(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle\)$:  
  $$max_{\alpha}\;\sum_{i}\alpha_i \;-\;\tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j\,y_i y_j\,K(x_i,x_j)
  $$
  subject to \$(0 \le \alpha_i \le C\)$ and \$(\sum_i \alpha_i y_i = 0\)$.

---

### 🔑 Support Vectors
- **Definition**: Training samples with non‑zero Lagrange multipliers \(\alpha_i>0\).  
- **Role**: They lie **on or inside** the margin boundaries  
  \$(\bigl(y_i(\mathbf{w}^\top x_i + b)=1-\xi_i\bigr)\)$ and **define** the optimal hyperplane.

---

### 3. Prediction
For a new point \(x\), compute the decision function:
$$hat{y} = \text{sign}\!\Bigl(\sum_{i}\alpha_i\,y_i\,K(x_i, x)\;+\;b\Bigr).
$$

Support vectors are the only points that contribute to this sum, making SVMs both **efficient** and **robust**.  


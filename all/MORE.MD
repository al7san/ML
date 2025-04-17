## ➗ Linear Regression Algorithm

### 📖 Definition  
Linear regression fits a model of the form  
$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \varepsilon_i
$$  
by estimating coefficients $\beta = [\beta_0,\dots,\beta_p]$ that **minimize the sum of squared residuals**.

---

### 🔢 Steps to Apply Linear Regression & Compute R‑squared / RMSE

1. **Data Preparation**  
   - Handle missing values (imputation or removal).  
   - Encode categorical variables (one‑hot, ordinal).  
   - (Optional) Standardize or normalize features if scales vary.

2. **Fit the Model**  
   - **Closed form** (Normal Equation):  
     $$hat\beta = (X^\top X)^{-1} X^\top y
     $$  
   - **Iterative** (Gradient Descent):  
     Update $beta \leftarrow \beta - \alpha\,\nabla_{\beta} \bigl\|X\beta - y\bigr\|^2$ until convergence.

3. **Make Predictions**  
   - Compute fitted values:  
     $$hat y = X\,\hat\beta.
     $$

4. **Compute Residuals**  
   - For each sample:  
     $$e_i = y_i - \hat y_i.
     $$

5. **Calculate R‑squared**  
   Measures the proportion of variance explained:  
   $$R^2
   = 1 \;-\;
   \frac{\sum_{i=1}^N (y_i - \hat y_i)^2}
        {\sum_{i=1}^N (y_i - \bar y)^2}
   \,,\quad
   \bar y = \frac{1}{N}\sum_{i=1}^N y_i.
   $$

6. **Calculate RMSE**  
   The root mean squared error:  
   $$mathrm{RMSE}
   = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat y_i)^2}.
   $$

7. **Evaluate & Diagnose**  
   - **Residual plots**: check for non‑random patterns (linearity, homoscedasticity).  
   - **QQ‑plot**: assess normality of residuals.  
   - **Outliers / Influential points**: identify via Cook’s distance or leverage.  

---

🎯 **Summary**  
Follow these steps to train a linear model, predict, and quantify its performance via $R^2$ (explained variance) and RMSE (average error magnitude).

---
## 🔢 Statistical Concepts

---

### 🔹 Difference between Classification and Regression

| Aspect               | Classification                              | Regression                                        |
|----------------------|---------------------------------------------|---------------------------------------------------|
| **Target Variable**  | Discrete categories (classes)               | Continuous numeric values                         |
| **Example Task**     | Spam vs. Ham email prediction               | Predicting house prices                           |
| **Model Output**     | Class labels (e.g. “cat”, “dog”)            | Real‑valued estimate (e.g. \$350{,}000\$)         |
| **Typical Loss**     | Cross‑entropy, hinge loss                   | Mean Squared Error (MSE), Mean Absolute Error     |
| **Evaluation**       | Accuracy, precision/recall, F1‑score        | \$R^2\$, RMSE, MAE                                |

---

### 🔹 What Does **Maximum Likelihood Estimation** Mean?

**Maximum Likelihood Estimation (MLE)** is a method for estimating the parameters \(\theta\) of a statistical model. You choose \(\theta\) to **maximize** the probability (likelihood) of observing your data under that model.

1. **Likelihood Function**  
   Given data \(\mathcal{D} = \{x_1, x_2, \dots, x_N\}\) and a parametric model \(p(x \mid \theta)\), define the likelihood:
   $$(\theta)
   = P(\mathcal{D} \mid \theta)
   = \prod_{i=1}^{N} p(x_i \mid \theta).
   $$

2. **Log‑Likelihood**  
   It’s more convenient to maximize the log of the likelihood:
   $$ell(\theta)
   = \log L(\theta)
   = \sum_{i=1}^{N} \log p(x_i \mid \theta).
   $$

3. **MLE Estimate**  
   The MLE of \(\theta\) is
   $$hat{\theta}
   = \arg\max_{\theta}\,L(\theta)
   = \arg\max_{\theta}\,\ell(\theta).
   $$

4. **Example (Gaussian Mean)**  
   If \$(x_i \sim \mathcal{N}(\mu, \sigma^2)\)$ with known \(\sigma^2\), the log‑likelihood is
   $$ell(\mu)
   = -\frac{N}{2}\log(2\pi\sigma^2)
     -\frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2.
   $$
   Maximizing this w.r.t. \(\mu\) yields
   $$hat{\mu}
   = \frac{1}{N}\sum_{i=1}^N x_i
   $$  
   — the sample mean.

---

🎯 **Takeaway**  
MLE finds the parameter values that make the observed data most probable under your chosen model.  

----
## 🔄 Expectation–Maximization (EM) & Its Connection to Maximum Likelihood

### 📖 What Is Expectation–Maximization?

The **Expectation–Maximization (EM)** algorithm is an iterative method for finding **maximum likelihood estimates** of parameters in statistical models that depend on **unobserved (latent) variables**. EM alternates between:

1. **E‑Step (Expectation)**  
   Compute the expected value of the **complete‑data log‑likelihood**, using the current parameter estimate \(\theta^{(t)}\):  
   $$Q\bigl(\theta \mid \theta^{(t)}\bigr)
   = \mathbb{E}_{Z\mid X,\theta^{(t)}}\bigl[\log p(X,Z\mid \theta)\bigr],
   $$  
   where \(X\) is the observed data and \(Z\) are the latent variables.

2. **M‑Step (Maximization)**  
   Maximize this expectation to update the parameters:  
   $$theta^{(t+1)}
   = \arg\max_{\theta}\;Q\bigl(\theta \mid \theta^{(t)}\bigr).
   $$

These two steps are repeated until convergence (i.e., until \(\theta^{(t+1)}\approx \theta^{(t)}\)).

---

### 🎯 How EM Relates to Maximum Likelihood

- **Objective**: EM seeks to **maximize the observed‑data log‑likelihood**  
  $$ell(\theta)
  = \log p(X \mid \theta)
  = \log \int p(X,Z \mid \theta)\,dZ.
  $$
- **Challenge**: Direct maximization may be intractable when the integral over \(Z\) is complex.
- **EM Solution**: By introducing the auxiliary function \(Q(\theta\mid\theta^{(t)})\), EM guarantees that each iteration **does not decrease** the observed‑data likelihood:
  $$ell\bigl(\theta^{(t+1)}\bigr)
  \;\ge\;
  \ell\bigl(\theta^{(t)}\bigr).
  $$
- In effect, EM finds a sequence of parameter estimates that **monotonically increase** the likelihood, converging to a (local) maximum of the likelihood function.

---

### 📌 Example: Gaussian Mixture Model (GMM)

For a GMM with \(K\) components, latent \$(Z_i\in\{1,\dots,K\}\)$:

- **E‑Step**: Compute responsibilities  
  $$gamma_{ik}
  = P\bigl(Z_i = k \mid x_i,\theta^{(t)}\bigr).
  $$
- **M‑Step**: Update mixing weights \(\pi_k\), means \$(\mu_k\)$, and covariances \$(\Sigma_k\)$ using \$(\{\gamma_{ik}\}\$).

This iteratively maximizes the mixture model’s likelihood, which would be hard to optimize directly.

---

> **Bottom Line**:  
> EM is a practical, general‑purpose algorithm for **MLE** in models with latent variables, turning an intractable optimization into two simpler steps that guarantee non‑decreasing likelihood.  


---
## 🛠️ Improving a Poorly Performing Model

When a model yields unsatisfactory results, follow these systematic steps:

1. **Diagnose the Issue**  
   - Inspect **training vs. validation** performance to see underfitting vs. overfitting.  
   - Plot **residuals** (regression) or **confusion matrix** (classification) to identify patterns or biases.

2. **Examine the Data**  
   - Check for **missing**, **outlier**, or **erroneous** values and clean or impute appropriately.  
   - Ensure **class balance** (for classification) or consider **target transformation** (for skewed regression).

3. **Feature Engineering**  
   - Create **new features** (interactions, aggregates, polynomial terms).  
   - Remove or combine **redundant** or **irrelevant** features.  
   - Apply **encoding** (one‑hot, ordinal) and **scaling** (standardization, normalization) as needed.

4. **Model Complexity & Choice**  
   - If underfitting: try a **more complex** model or reduce regularization.  
   - If overfitting: simplify the model (prune, reduce layers/trees), increase regularization, or add dropout.

5. **Hyperparameter Tuning**  
   - Use **Grid Search**, **Random Search**, or **Bayesian Optimization** with **cross‑validation** to find optimal settings.

6. **Ensemble Methods**  
   - Combine multiple models (bagging, boosting, stacking) to reduce variance and improve generalization.

7. **Data Augmentation & More Data**  
   - Expand the training set via **augmentation** (images, text) or by **collecting additional samples**.  
   - Use **SMOTE** or similar techniques for imbalanced classes.

8. **Regularization & Early Stopping**  
   - Apply **L1/L2 penalties**, **dropout**, or **early stopping** (monitor validation loss) to prevent overfitting.

9. **Cross‑Validation & Robust Evaluation**  
   - Use **k‑fold** or **stratified** splits to ensure stability of performance estimates.  
   - Evaluate with multiple metrics (accuracy, F1, RMSE, AUC) relevant to your task.

10. **Iterate & Validate**  
    - Implement one change at a time, compare metrics, and keep only those that improve performance.  
    - Maintain a **tracking log** (e.g., experiment notebook) to document what works.

---

🎯 **Goal**: Through careful data preparation, model selection, tuning, and validation, transform a poor model into one that generalizes reliably to new data.  

## Bias vs. Variance in Machine Learning

### 🔹 Bias
- **Definition**: Bias is the error introduced by approximating a real-world problem (which may be very complex) with a simplified model.  
- **High Bias**: The model makes strong assumptions about the data, often missing important patterns.  
- **Effect**: Leads to **underfitting** — the model is too simple to capture the data's complexity.  
- **Example**: Using a linear model to fit non-linear data.

### 🔹 Variance
- **Definition**: Variance is the error introduced by the model's sensitivity to small fluctuations in the training data.  
- **High Variance**: The model learns noise in the training data as if it were a real pattern.  
- **Effect**: Leads to **overfitting** — the model performs well on training data but poorly on new, unseen data.  
- **Example**: A very deep decision tree that memorizes the training data.

---

### 🎯 Bias-Variance Tradeoff
- There’s a tradeoff between **bias** and **variance**:
  - A **simple model** has **high bias** and **low variance**.
  - A **complex model** has **low bias** and **high variance**.
- ✅ The goal is to find the **sweet spot** where both bias and variance are balanced to minimize total error (**generalization error**).

This diagram shows how model complexity affects bias and variance:

![Bias-Variance Tradeoff]()

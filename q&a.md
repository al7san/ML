## Scenario:
A school suspects some students are cheating during exams. The school collects data such as:
- Test start and end time
- Answers similarity between students
- Seating arrangements
- Mouse/keyboard activity (for online exams)
- Sudden spikes in performance
- Historical grades

We aim to use machine learning to identify students with suspicious behavior.

---

## 1. Understanding the Problem Type

Cheating detection can be modeled in two ways:
- **Supervised learning**: if we have labeled data indicating who cheated (e.g., confirmed cases).
- **Unsupervised learning**: if no labels are available, and we want to detect patterns or outliers.

---

## 2. Suitable Algorithms and Why We Choose Them

### A. Decision Tree
- **Why choose it**:
  - Easy to interpret; decisions can be explained to school authorities.
  - Works well with both categorical and numerical data.
- **Why not others here**:
  - Logistic Regression gives probabilities but not clear logic.
  - Neural Networks are black-box models.

### B. Random Forest
- **Why choose it**:
  - More robust than a single decision tree.
  - Reduces overfitting and shows feature importance.
- **Why not only Decision Tree**:
  - Trees can overfit; forests are more stable.

### C. K-Means (Unsupervised Clustering)
- **Why choose it**:
  - Useful when labels are unavailable.
  - Can cluster similar behaviors â potential cheating groups.
- **Why not supervised models**:
  - Need labeled data.

### D. Isolation Forest
- **Why choose it**:
  - Designed for anomaly detection.
  - Effective in spotting rare, extreme behavior.
- **Why not only K-Means**:
  - K-Means finds clusters, not anomalies.

### E. Neural Networks
- **Why choose it** (only if):
  - Dataset is large and complex.
- **Why not by default**:
  - Hard to interpret; needs high computation.

### F. Support Vector Machines (SVM)
- **Why choose it**:
  - Strong in small/medium datasets and high dimensions.
- **Why not always**:
  - Hard to scale and explain.

### G. Logistic Regression
- **Why choose it**:
  - Simple and interpretable.
- **Why not only use it**:
  - Misses complex relationships.

### H. Naive Bayes
- **Why choose it**:
  - Fast and simple.
- **Why not use it**:
  - Assumes feature independence (rare in this case).

---

## 3. Example Approach Based on Data

### If labels exist:
- Use Decision Tree or Random Forest.
- Evaluate with Confusion Matrix, Precision, Recall.
- For complex data, try SVM or Neural Networks.

### If no labels:
- Use K-Means for clustering.
- Use Isolation Forest for anomalies.
- Review flagged cases manually.

---

## 4. Summary Table

| Algorithm           | Use Case                  | Pros                                   | Cons                                |
|---------------------|---------------------------|----------------------------------------|-------------------------------------|
| Decision Tree        | Labeled data              | Interpretable, rule-based              | Prone to overfitting                |
| Random Forest        | Labeled data              | Accurate, robust, feature importance   | Slower, less interpretable          |
| K-Means              | Unlabeled data            | Finds patterns                         | Needs manual interpretation         |
| Isolation Forest     | Unlabeled data (anomaly)  | Good for outlier detection             | May flag false positives            |
| Neural Network       | Complex, large data       | Captures nonlinear behavior            | Black-box, less explainable         |
| SVM                  | Labeled, small data       | Effective in high dimensions           | Hard to scale, less transparent     |
| Logistic Regression  | Simple classification     | Fast, interpretable                    | Misses nonlinear relations          |
| Naive Bayes          | Categorical features      | Fast, simple                           | Unrealistic independence assumption |

---

## 5. Conclusion

For cheating detection:
- Use **Decision Trees/Random Forests** when interpretability is key.
- Use **K-Means/Isolation Forest** when exploring unlabeled data.
- Avoid complex models unless needed and interpretable.

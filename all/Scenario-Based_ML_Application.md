# Scenario-Based ML Application: Detecting Cheating in Exams

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
  - **Interpretability**: A decision tree provides a set of clear, interpretable rules, which is helpful when explaining results to school authorities.
  - Works well with both **categorical** and **numerical data**.
- **Why not others here**:
  - **Logistic Regression** gives probabilities but doesn’t provide clear logical steps.
  - **Neural Networks** are considered black-box models and are harder to explain.

### B. Random Forest
- **Why choose it**:
  - **Robustness**: A random forest is an ensemble of decision trees, making it less prone to overfitting compared to a single decision tree.
  - Provides **feature importance**, which can help understand which factors influence cheating detection the most.
- **Why not only Decision Tree**:
  - A single decision tree may overfit the data, whereas a random forest averages multiple trees to reduce that risk.

### C. K-Means (Unsupervised Clustering)
- **Why choose it**:
  - **No labeled data**: If no labeled data is available, K-Means can help cluster students based on behaviors, allowing us to spot potential outliers or groups that exhibit similar cheating patterns.
  - Helps to find **similarities** in behavior, potentially grouping students who show similar suspicious activity.
- **Why not supervised models**:
  - Supervised models (like Decision Trees) require labeled data, which may not be available for all students.

### D. Isolation Forest
- **Why choose it**:
  - **Anomaly detection**: Specifically designed for anomaly detection, Isolation Forest works well to identify rare, unusual behaviors, such as students who are cheating.
  - It works by **isolating anomalies** rather than profiling normal behavior, making it efficient in spotting outliers.
- **Why not only K-Means**:
  - K-Means is good for clustering but doesn’t specialize in detecting anomalies, which is crucial for identifying rare behaviors (e.g., sudden performance spikes).

### E. Neural Networks
- **Why choose it** (only if):
  - **Large and complex data**: Neural networks perform well on large datasets with complex patterns, which might arise in more detailed exam datasets.
  - **Flexibility**: Neural networks can capture complex, nonlinear relationships that simpler models may miss.
- **Why not by default**:
  - **Interpretability**: Neural networks are black-box models, which makes them difficult to explain to non-technical users like school administrators.
  - They also require significant **computational resources** and a large dataset to perform optimally.

### F. Support Vector Machines (SVM)
- **Why choose it**:
  - **Effective in high-dimensional spaces**: SVM works well when the data has many features (e.g., multiple exam-related factors), which is common in complex cheating detection.
  - **Margin maximization**: SVM attempts to find the optimal hyperplane that best separates data points into distinct categories.
- **Why not always**:
  - **Scalability**: SVMs are not ideal for very large datasets due to computational costs.
  - **Interpretability**: SVMs are harder to interpret compared to decision trees.

### G. Logistic Regression
- **Why choose it**:
  - **Simplicity and interpretability**: It is easy to implement and interpret, providing a probabilistic approach to classify students as potentially cheating or not.
  - **Quick to train**: It requires less computational power compared to more complex algorithms like Neural Networks.
- **Why not only use it**:
  - **Linear relationships**: Logistic regression assumes linear relationships between variables, which may not capture complex patterns in cheating behavior.

### H. Naive Bayes
- **Why choose it**:
  - **Simplicity and speed**: Naive Bayes is fast and works well with categorical data. It can be used in cases where quick predictions are required.
- **Why not use it**:
  - **Independence assumption**: Naive Bayes assumes that all features are independent, which is rarely the case in cheating scenarios (e.g., students with similar exam-taking behaviors are likely correlated).

---

## 3. Example Approach Based on Data

### If labels exist:
- **Supervised approach**: Use **Decision Tree** or **Random Forest** to classify students as likely to have cheated or not. These models can be evaluated using performance metrics like **Confusion Matrix**, **Precision**, and **Recall**.
- If more complexity is needed, try **SVM** or **Neural Networks**, especially if the data involves more intricate relationships.

### If no labels:
- **Unsupervised approach**: Use **K-Means** for clustering to detect groups of students with similar suspicious behavior.
- **Isolation Forest** can be used to detect anomalies, highlighting students whose performance or behavior deviates significantly from the norm.
- After detection, manually review the flagged students for further investigation.

---

## 4. Summary Table

| Algorithm           | Use Case                  | Pros                                   | Cons                                |
|---------------------|---------------------------|----------------------------------------|-------------------------------------|
| Decision Tree        | Labeled data              | Easy to interpret, rule-based logic    | Prone to overfitting                |
| Random Forest        | Labeled data              | Accurate, robust, shows feature importance | Slower, less interpretable          |
| K-Means              | Unlabeled data            | Finds patterns in data                 | Requires manual review of clusters  |
| Isolation Forest     | Unlabeled data (anomaly)  | Good for detecting rare, extreme behavior | May flag false positives            |
| Neural Network       | Large and complex data    | Captures nonlinear relationships       | Black-box, hard to interpret        |
| SVM                  | Small to medium labeled data | Works well in high-dimensional space   | Not easily interpretable, slow with large datasets |
| Logistic Regression  | Simple classification     | Fast, easy to interpret                | Assumes linear relationships        |
| Naive Bayes          | Categorical features      | Fast, simple, works well with text data | Assumes feature independence        |

---

## 5. Conclusion

For cheating detection:
- Use **Decision Trees** or **Random Forests** when interpretability and understanding feature importance are crucial.
- Use **K-Means** and **Isolation Forest** if data labels are unavailable, and you're focusing on detecting clusters or anomalies.
- Avoid **complex models** like Neural Networks unless the dataset is large and you need to capture complex, non-linear relationships, but be aware of the lack of interpretability.

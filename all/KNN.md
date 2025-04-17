## 🔍 K‑Nearest Neighbors (KNN) Algorithm

### 📖 Definition  
KNN is a **non‑parametric**, instance‑based learning algorithm used for classification and regression. It predicts the label of a new point by looking at the labels of its $K$ nearest neighbors in feature space.

---

### ✅ Advantages
- **Simplicity**: Intuitive and easy to implement.  
- **No Training Phase**: “Lazy” learning—stores the dataset and delays computation until prediction.  
- **Adaptable to Multi‑Class**: Naturally handles more than two classes without modification.  
- **Flexible Distance Metrics**: You can choose Euclidean, Manhattan, Minkowski, etc.

### ⚠️ Disadvantages
- **Computational Cost at Prediction**: Distance calculations over the entire dataset for each query can be slow for large $N$.  
- **Storage Cost**: Must store the entire training set.  
- **Curse of Dimensionality**: Distance measures become less meaningful as the number of features grows.  
- **Choice of $K$ and Metric**: Performance sensitive to the choice of $K$ and distance metric.  
- **Imbalanced Data**: May be biased toward the majority class if one class dominates.

---

### 🔄 How to Use KNN

1. **Prepare the Data**  
   - **Clean** and **impute** missing values.  
   - **Scale/Normalize** features so that distances are meaningful (e.g., standardization).

2. **Choose a Distance Metric**  
   - **Euclidean** (most common):  
     $$d(x, x_i)
     = \sqrt{\sum_{j=1}^{n} (x_j - x_{i,j})^2}
     $$
   - **Manhattan**: $d(x,x_i)=\sum_j |x_j - x_{i,j}|$, etc.

3. **Select $K$**  
   - Use **cross‑validation** to try different $K$ values and pick the one with the best validation score.

4. **Prediction for a New Sample $x$**  
   1. Compute distances $d(x, x_i)$ to all training points.  
   2. **Sort** neighbors by increasing distance and pick the top $K$.  
   3. **Classification**:  
      $$hat y = \text{mode}\bigl(\{\,y_i : x_i \in \text{neighbors}\}\bigr)
      $$
   4. **Regression**:  
      $$hat y = \frac{1}{K}\sum_{x_i \in \text{neighbors}} y_i
      $$

5. **Evaluate & Tune**  
   - Assess performance (accuracy, RMSE, etc.) via cross‑validation.  
   - Experiment with different $K$, distance metrics, and feature weighting schemes.

---

🎯 **Tip**:  
For large datasets, consider approximate nearest‑neighbor methods (e.g., KD‑trees, Ball trees, or Locality‑Sensitive Hashing) to speed up queries.  

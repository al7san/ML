

## **1. K-Means Clustering Algorithm**  
### **Full Explanation**  
**What is it?**  
An **unsupervised** algorithm that groups similar data points into clusters based on their features.  

**How it works:**  
1. **Choose `k` clusters** (e.g., 2 or 3).  
2. **Randomly initialize centroids** (cluster centers).  
3. **Assign each point to the nearest centroid** (using Euclidean distance).  
4. **Recalculate centroids** as the mean of points in each cluster.  
5. **Repeat** until centroids stabilize.  

### **Key Concepts**  
1. **Centroid**  
   - The "center" of a cluster (mean of all points in it).  
   - Updated iteratively to minimize within-cluster distance.  

2. **Euclidean Distance**  
   - Measures distance between two points:  
     \[
     d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
     \]  

3. **Elbow Method**  
   - Technique to find the optimal `k`:  
     - Plot `k` vs. **sum of squared errors (SSE)**.  
     - Pick `k` where the curve bends ("elbow").  

### **Real-World Example**  
**Scenario**: A retail store segments customers into 3 groups based on:  
- **Monthly spending**.  
- **Website visits**.  

**Resulting Clusters**:  
1. **Low spenders** (rare purchases).  
2. **Medium spenders**.  
3. **High spenders** (frequent purchases).  

### **Mathematical Example**  
**Data**: 4 points in 2D space:  
- \( A(1,1) \), \( B(1,2) \), \( C(10,10) \), \( D(10,11) \).  

**Steps**:  
1. Set `k=2`; initial centroids: \( (1,1) \) and \( (10,10) \).  
2. Assign points:  
   - Cluster 1: \( A, B \) → New centroid: \( (1, 1.5) \).  
   - Cluster 2: \( C, D \) → New centroid: \( (10, 10.5) \).  

**Final Clusters**:  
- **Group 1**: \( A, B \).  
- **Group 2**: \( C, D \).  

---

## **2. Naive Bayes Classifier**  
### **Full Explanation**  
**What is it?**  
A **supervised** algorithm for classification based on **Bayes’ Theorem**, assuming feature independence ("naive").  

**How it works:**  
1. Calculate **class probability** \( P(C) \).  
2. Compute **feature probabilities** \( P(x_i \mid C) \).  
3. Predict the class with the highest \( P(C \mid x_1, ..., x_n) \).  

### **Key Concepts**  
1. **Bayes’ Theorem**:  
   \[
   P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}
   \]  

2. **Class Prior (\( P(C) \))**:  
   - Fraction of samples in class \( C \).  

3. **Likelihood (\( P(x_i \mid C) \))**:  
   - Probability of feature \( x_i \) occurring in class \( C \).  

### **Real-World Example**  
**Scenario**: Classify emails as **"spam"** or **"not spam"** using keywords:  
- **Training Data**: 100 emails (60 spam, 40 not spam).  
  - Spam emails: 45 contain "free," 30 contain "win."  
  - Non-spam: 10 contain "free," 5 contain "win."  

**Prediction for "free win"**:  
1. \( P(\text{spam}) = 0.6 \).  
2. \( P(\text{"free"} \mid \text{spam}) = 0.75 \).  
3. \( P(\text{"win"} \mid \text{spam}) = 0.5 \).  
4. \( P(\text{spam} \mid \text{"free win"}) \propto 0.6 \times 0.75 \times 0.5 = 0.225 \).  
5. \( P(\text{not spam} \mid \text{"free win"}) \propto 0.4 \times 0.25 \times 0.125 = 0.0125 \).  

**Result**: Classified as **spam** (0.225 > 0.0125).  

### **Mathematical Example**  
**Training Data**:  
| Email | "free" | "win" | Class  |  
|-------|--------|-------|--------|  
| 1     | Yes    | No    | Spam   |  
| 2     | No     | Yes   | Spam   |  
| 3     | Yes    | No    | Not Spam |  

**Calculations**:  
- \( P(\text{spam}) = \frac{2}{3} \).  
- \( P(\text{"free"} \mid \text{spam}) = \frac{1}{2} \).  
- \( P(\text{"win"} \mid \text{spam}) = \frac{1}{2} \).  

---

## **3. Support Vector Machine (SVM)**  
### **Full Explanation**  
**What is it?**  
A **supervised** algorithm for classification/regression that finds the **optimal hyperplane** separating classes.  

**How it works:**  
1. Identify **support vectors** (closest points to the hyperplane).  
2. Maximize the **margin** (distance between classes).  

### **Key Concepts**  
1. **Hyperplane**:  
   - Decision boundary: \( w \cdot x + b = 0 \).  
   - \( w \): Weight vector.  
   - \( b \): Bias term.  

2. **Support Vectors**:  
   - Critical points that define the margin.  

3. **Kernel Trick**:  
   - Transforms non-linear data into higher dimensions (e.g., RBF kernel).  

### **Real-World Example**  
**Scenario**: Classify images as **cats** vs. **dogs** using features:  
- Ear shape.  
- Eye color.  

**SVM Action**:  
- Finds a hyperplane separating cat/dog features in high-dimensional space.  

### **Mathematical Example**  
**Data**:  
- Class +1: \( (1,1) \), \( (2,2) \).  
- Class -1: \( (0,0) \), \( (-1,-1) \).  

**Optimal Hyperplane**:  
- \( x - y = 0 \).  
- **Support Vectors**: \( (1,1) \) and \( (0,0) \).  

---

### **Quick Summary**  
| Algorithm     | Type         | Key Concepts                  | Real-World Use          |  
|--------------|--------------|-------------------------------|-------------------------|  
| **K-Means**  | Unsupervised | Centroids, Euclidean distance | Customer segmentation   |  
| **Naive Bayes** | Supervised | Bayes’ Theorem, Likelihood    | Spam filtering          |  
| **SVM**      | Supervised   | Hyperplane, Support Vectors   | Image classification    |  

--- 


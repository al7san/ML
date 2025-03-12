# **How to Determine the Root Node in a Decision Tree?**  
The **root node** in a Decision Tree is the feature that provides the best split of the data. It is selected based on mathematical criteria that measure the **importance of each feature** in classification or regression tasks.  

---

## **Methods for Selecting the Root Node**  

### **1️⃣ Gini Impurity – Commonly Used in Classification (CART Algorithm)**  
- Measures **how impure** a node is (i.e., how mixed the classes are).  
- The feature that results in the **lowest Gini Impurity** after splitting is selected as the root.  
- **Formula:**  
  \[
  Gini = 1 - \sum p_i^2
  \]
  where \( p_i \) is the probability of each class in the node.  

✅ **Example:** If you are building a decision tree to classify customers based on age and income, the feature that reduces **Gini Impurity** the most will be the **root node**.  

---

### **2️⃣ Information Gain – Used in ID3 Algorithm**  
- Based on **Entropy**, which measures disorder in the data.  
- The feature that results in the **greatest reduction in Entropy** is chosen as the root.  
- **Formula for Entropy:**  
  \[
  Entropy = -\sum p_i \log_2 p_i
  \]
- **Information Gain (IG)** is calculated as:  
  \[
  IG = Entropy_{before} - Entropy_{after}
  \]

✅ **Example:** If you are predicting whether a person will buy a product based on age and gender, the feature that gives the **highest Information Gain** is chosen as the root.  

---

### **3️⃣ Reduction in Variance – Used in Regression Trees**  
- In Regression Trees, the goal is to **minimize variance** in numerical values after splitting.  
- The feature that leads to the **largest reduction in variance** is chosen as the root.  

✅ **Example:** If you are predicting house prices based on square footage and number of rooms, the feature that minimizes **variance in price** after splitting will be the root.  

---

## **💡 Which Method Should You Use?**  
- **For Classification Trees → Use Gini Impurity or Information Gain.**  
- **For Regression Trees → Use Reduction in Variance.**

---
# **Support Vector Machine (SVM) – Overview**  

## **What is SVM?**  
Support Vector Machine (SVM) is a **supervised learning algorithm** used for **classification** and **regression** tasks. It works by finding the **optimal decision boundary** (hyperplane) that best separates data points into different classes.  

---

## **How Does SVM Work?**  

### **1️⃣ Finding the Optimal Hyperplane**  
- In **binary classification**, SVM finds a **hyperplane** (a line in 2D, a plane in 3D, or a higher-dimensional surface) that maximizes the **margin** between two classes.  
- The margin is the distance between the hyperplane and the **closest data points** (called **support vectors**).  
- The larger the margin, the better the model generalizes to new data.  

### **2️⃣ Handling Non-Linearly Separable Data – The Kernel Trick**  
- If the data is **not linearly separable**, SVM uses a **kernel function** to transform data into a higher-dimensional space where it becomes linearly separable.  
- Common kernel functions:  
  - **Linear Kernel**: Suitable for linearly separable data.  
  - **Polynomial Kernel**: Maps data to a higher polynomial degree.  
  - **Radial Basis Function (RBF) Kernel**: Captures complex relationships by mapping data to infinite dimensions.  
  - **Sigmoid Kernel**: Similar to neural networks.  

---

## **Mathematical Formulation**  
The decision boundary is defined as:  

\[
f(x) = w^T x + b
\]

where:  
- \( w \) is the weight vector,  
- \( x \) is the input feature vector,  
- \( b \) is the bias term.  

The optimization goal is to:  

\[
\min ||w||^2
\]

subject to:  

\[
y_i (w^T x_i + b) \geq 1, \quad \forall i
\]

where \( y_i \) represents the class labels (\(+1\) or \(-1\)).  

For non-linearly separable data, we introduce a **slack variable** \( \xi_i \) to allow some misclassification, leading to **soft-margin SVM**:  

\[
\min \frac{1}{2} ||w||^2 + C \sum \xi_i
\]

where **C** controls the trade-off between margin maximization and classification error.  

---

## **Applications of SVM**
✔️ **Text Classification**: Spam detection, sentiment analysis.  
✔️ **Image Recognition**: Face detection, handwriting recognition.  
✔️ **Bioinformatics**: Cancer classification from gene expression data.  
✔️ **Finance**: Credit risk assessment, stock price prediction.  

---

## **Pros & Cons of SVM**
### ✅ **Advantages**  
- Effective in **high-dimensional spaces**.  
- Works well for **small to medium-sized datasets**.  
- Can handle **non-linearly separable** data using **kernel tricks**.  

### ❌ **Disadvantages**  
- Computationally expensive for **large datasets**.  
- Requires careful tuning of **hyperparameters (C, kernel type, gamma, etc.)**.  
- Difficult to interpret compared to decision trees.  

---

  

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

# Confusion Matrix Overview

The **Confusion Matrix** is a table used to evaluate the performance of classification models. It shows the actual vs predicted classifications, providing insights into various evaluation metrics.

## Confusion Matrix

|                       | **Actual Positive** | **Actual Negative** |
|-----------------------|---------------------|---------------------|
| **Predicted Positive** | True Positive (TP)   | False Positive (FP) |
| **Predicted Negative** | False Negative (FN)  | True Negative (TN)  |

### Key Terms:
- **True Positive (TP):** The model correctly predicted a positive outcome (predicted positive and actual positive).
- **False Positive (FP):** The model incorrectly predicted a positive outcome (predicted positive and actual negative).
- **False Negative (FN):** The model incorrectly predicted a negative outcome (predicted negative and actual positive).
- **True Negative (TN):** The model correctly predicted a negative outcome (predicted negative and actual negative).

---

## Properties of Confusion Matrix

### 1. **Precision**
   - **Definition:** Precision is the ratio of correctly predicted positive observations to the total predicted positives.
   - **Formula:**  
     \[
     \text{Precision} = \frac{TP}{TP + FP}
     \]
   - **Interpretation:** Precision answers the question: "How many of the instances predicted as positive were actually positive?"

### 2. **Recall (Sensitivity, True Positive Rate)**
   - **Definition:** Recall is the ratio of correctly predicted positive observations to all the actual positives.
   - **Formula:**  
     \[
     \text{Recall} = \frac{TP}{TP + FN}
     \]
   - **Interpretation:** Recall answers the question: "How many of the actual positives were correctly predicted?"

### 3. **Accuracy**
   - **Definition:** Accuracy is the ratio of correctly predicted observations (both positive and negative) to the total observations.
   - **Formula:**  
     \[
     \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
     \]
   - **Interpretation:** Accuracy answers the question: "How many predictions were correct overall?"

### 4. **F1 Score**
   - **Definition:** The F1 Score is the harmonic mean of Precision and Recall. It is useful when you need a balance between Precision and Recall.
   - **Formula:**  
     \[
     F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
     \]
   - **Interpretation:** The F1 Score balances the trade-off between Precision and Recall, especially when you need to balance false positives and false negatives.

### 5. **Specificity (True Negative Rate)**
   - **Definition:** Specificity is the ratio of correctly predicted negative observations to all the actual negatives.
   - **Formula:**  
     \[
     \text{Specificity} = \frac{TN}{TN + FP}
     \]
   - **Interpretation:** Specificity answers the question: "How many of the actual negatives were correctly predicted?"

### 6. **False Positive Rate (FPR)**
   - **Definition:** FPR is the ratio of incorrectly predicted negative observations to all actual negatives.
   - **Formula:**  
     \[
     \text{FPR} = \frac{FP}{FP + TN}
     \]
   - **Interpretation:** FPR answers the question: "How many of the actual negatives were incorrectly predicted as positives?"

### 7. **False Negative Rate (FNR)**
   - **Definition:** FNR is the ratio of incorrectly predicted positive observations to all actual positives.
   - **Formula:**  
     \[
     \text{FNR} = \frac{FN}{FN + TP}
     \]
   - **Interpretation:** FNR answers the question: "How many of the actual positives were incorrectly predicted as negatives?"

---

## Example Scenario

Consider a medical test where we want to predict whether a patient has a disease (positive) or not (negative). Let's assume the following:

- **True Positive (TP):** Patients who actually have the disease and the model correctly predicts them as positive.
- **False Positive (FP):** Patients who do not have the disease but the model incorrectly predicts them as positive.
- **False Negative (FN):** Patients who actually have the disease but the model incorrectly predicts them as negative.
- **True Negative (TN):** Patients who do not have the disease and the model correctly predicts them as negative.

---

## Conclusion

The confusion matrix helps to evaluate a classification model by providing detailed information about the types of errors it makes. By understanding the components of the confusion matrix, such as Precision, Recall, and Accuracy, you can make more informed decisions about how well the model is performing and where improvements can be made.

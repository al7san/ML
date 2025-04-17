# Machine Learning Cheat Sheet

A concise overview of core Machine Learning concepts, algorithms, and practical tips—complete with real‑world examples.

---

## Table of Contents

1. [What is Machine Learning?](#what-is-machine-learning)  
2. [Bias vs. Variance](#bias-vs-variance)  
3. [Underfitting vs. Overfitting](#underfitting-vs-overfitting)  
4. [Confusion Matrix](#confusion-matrix)  
5. [Decision Trees](#decision-trees)  
6. [Neural Networks](#neural-networks)  
7. [K‑Means Clustering](#k-means-clustering)  
8. [Naive Bayes](#naive-bayes)  
9. [Support Vector Machines (SVM)](#support-vector-machines-svm)  
10. [K‑Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)  
11. [Linear Regression](#linear-regression)  
12. [Expectation–Maximization (EM)](#expectation–maximization-em)  
13. [Classification vs. Regression](#classification-vs-regression)  
14. [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)  
15. [Improving a Poorly Performing Model](#improving-a-poorly-performing-model)  

---

## What is Machine Learning?

Machine Learning (ML) enables computers to **learn patterns** and make **predictions or decisions** from data without explicit programming.  
- **Supervised Learning**: learn from labeled data (classification, regression)  
- **Unsupervised Learning**: find structure in unlabeled data (clustering, dimensionality reduction)  
- **Reinforcement Learning**: learn by interacting with an environment

---

## Bias vs. Variance

- **Bias**: Error from an overly simple model that misses key patterns.  
  *Example*: Fitting a straight line to seasonal sales data.  
- **Variance**: Error from an overly complex model that learns noise.  
  *Example*: A sales predictor that chases random daily spikes.

---

## Underfitting vs. Overfitting

- **Underfitting**: Model too simple → poor on training & test.  
  *Example*: Linear regression for mountainous terrain.  
- **Overfitting**: Model too complex → great on training, poor on new data.  
  *Example*: Decision tree memorizing holiday spikes.  
- **Prevention**:  
  - Underfit → increase complexity, add features  
  - Overfit → simplify model, regularize, gather/clean data

---

## Confusion Matrix

|               | Predicted + | Predicted – |
|---------------|-------------|-------------|
| **Actual +**  | TP          | FN          |
| **Actual –**  | FP          | TN          |

*Example*: Spam filter — count of missed spam (FN) and false alarms (FP).

---

## Decision Trees

- **How it works**: Recursively split data by feature tests to form a flowchart.  
  *Example*: Loan approval via income > \$50K → debt ratio < 40% → …  
- **Pros**: Interpretable, no feature scaling  
- **Cons**: Prone to overfitting, unstable

---

## Neural Networks

- **How it works**: Stacked layers of “neurons” learn hierarchical features.  
  *Example*: Face recognition on smartphones  
- **Pros**: Powerful on large, complex data  
- **Cons**: Data‑hungry, “black box,” compute‑intensive

---

## K‑Means Clustering

- **How it works**: Partition data into *K* clusters by minimizing distance to centroids.  
  *Example*: Segment customers by purchase frequency & basket size  
- **Pros**: Simple, fast  
- **Cons**: Must choose *K*, assumes spherical clusters, outliers skew results

---

## Naive Bayes

- **How it works**: Uses Bayes’ theorem with feature independence assumption.  
  *Example*: Classify news articles by word counts  
- **Pros**: Fast, works with small data, gives probabilities  
- **Cons**: Independence assumption is often false, zero‑count issues

---

## Support Vector Machines (SVM)

- **How it works**: Finds max‑margin hyperplane separating classes (linear or via kernels).  
  *Example*: Distinguish typed vs. handwritten signatures  
- **Pros**: Effective in high dimensions, robust generalization  
- **Cons**: Slow on large datasets, sensitive to hyperparameters

---

## K‑Nearest Neighbors (KNN)

- **How it works**: Classifies by majority vote among *K* closest training points.  
  *Example*: Movie recommendations based on similar user preferences  
- **Pros**: Intuitive, no training  
- **Cons**: Slow at inference, suffers in high dimensions

---

## Linear Regression

- **How it works**: Fits a line/hyperplane by minimizing squared errors.  
- **Key formulas**:  
  - $\hat\beta=(X^\top X)^{-1}X^\top y$  
  - $R^2 = 1 - \frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$  
  - $\mathrm{RMSE}=\sqrt{\tfrac{1}{N}\sum(y-\hat y)^2}$  
- *Example*: Predict house prices from size & location

---

## Expectation–Maximization (EM)

- **How it works**: Iterates E‑step (estimate latent variables) and M‑step (optimize parameters) to maximize likelihood in models with hidden variables.  
- *Example*: Gaussian Mixture Models for customer segmentation

---

## Classification vs. Regression

- **Classification**: Predicts discrete labels (e.g., spam vs. not spam)  
- **Regression**: Predicts continuous values (e.g., sales amount)

---

## Maximum Likelihood Estimation (MLE)

- **How it works**: Chooses parameters $\theta$ to maximize the likelihood $L(\theta)=\prod_i p(x_i\mid\theta)$ or equivalently $\ell(\theta)=\sum_i\log p(x_i\mid\theta)$.  
- *Example*: Estimating population mean under a normal model

---

## Improving a Poorly Performing Model

1. Diagnose under/overfitting via error plots & metrics  
2. Clean data: handle outliers & missing values  
3. Engineer/select better features  
4. Adjust model complexity or choose a new algorithm  
5. Tune hyperparameters (Grid/Random/Bayesian Search)  
6. Ensemble models or gather more data  

*Example*: Face‑recognition errors between identical twins → collect more varied images or use deeper network with dropout.

---

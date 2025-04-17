## 🌑 K‑Means Clustering Algorithm

### 📖 Definition  
K‑Means is an **unsupervised** clustering algorithm that partitions \(N\) data points \(\{x_i\}\) into \(K\) clusters \(\{C_1,\dots,C_K\}\) by minimizing the within‐cluster sum of squared distances to their centroids \(\{\mu_k\}\):

$$
\min_{C,\;\mu}\;
\sum_{k=1}^{K}
\sum_{x_i \,\in\,C_k}
\bigl\lVert x_i - \mu_k\bigr\rVert^2
\quad\text{where}\quad
\mu_k = \frac{1}{|C_k|}\sum_{x_i\in C_k} x_i
$$

**Basic steps**:
1. Initialize \(K\) centroids (randomly or via smart seeding like K‑Means++).  
2. **Assign** each point to its nearest centroid.  
3. **Recompute** each centroid as the mean of its assigned points.  
4. **Repeat** steps 2–3 until assignments no longer change (or a maximum number of iterations is reached).

---

### ✅ Advantages
- **Simplicity & Speed**  
  Straightforward to implement and very efficient on large datasets (time complexity roughly \(O(N\,K\,T)\) where \(T\) is the number of iterations).  
- **Scalability**  
  Scales to millions of samples if implemented with optimized libraries.  
- **Interpretability**  
  Easy to explain: clusters are “spherical” around centroids in feature space.  
- **Deterministic Structure**  
  Given fixed initialization, the algorithm’s behavior is consistent and reproducible.

---

### ⚠️ Disadvantages
- **Choice of \(K\)**  
  You must specify the number of clusters in advance; picking \(K\) often requires domain knowledge or methods like the elbow or silhouette analysis.  
- **Sensitivity to Initialization**  
  Poor initial centroids can lead to suboptimal clustering (can be mitigated by K‑Means++).  
- **Assumes Spherical Clusters**  
  Works best when clusters are roughly circular/convex and of similar size—struggles with elongated or variable‐density clusters.  
- **Outlier Sensitivity**  
  Outliers can skew centroids significantly because it uses the mean.  
- **Hard Assignment**  
  Each point belongs to exactly one cluster—no notion of “soft” membership.

---

🎯 ***Tip***:  
- Use **K‑Means++** for smarter initialization.  
- Evaluate multiple \(K\) values with the **elbow method** or **silhouette score**.  
- Consider **Gaussian Mixture Models** if clusters are non‑spherical or you need probabilistic assignments.  

---
## 🔄 How K‑Means Works & Centroid Calculation

K‑Means clusters $N$ data points $\{x_i\}_{i=1}^N$ into $K$ groups by iteratively repeating two main steps until convergence:

1. **Assignment Step**  
   Assign each point to the nearest centroid:  
   $$
   c_i \;=\;\arg\min_{k\in\{1,\dots,K\}}\;\bigl\lVert x_i - \mu_k\bigr\rVert^2
   $$
   where $c_i$ is the cluster index for point $x_i$ and $\{\mu_k\}$ are the current centroids.

2. **Update Step**  
   Recompute each centroid as the mean of all points assigned to it:  
   $$
   \mu_k
   = \frac{1}{|C_k|}\sum_{x_i \,\in\, C_k} x_i
   $$
   where $C_k = \{\,x_i : c_i = k\}$ and $|C_k|$ is the number of points in cluster $k$.

---

### 🔁 Full Algorithm Outline

1. **Initialize** centroids $\{\mu_k\}_{k=1}^K$ (randomly or with K‑Means++).  
2. **Repeat** until assignments don’t change or max iterations reached:  
   - **Assign** each $x_i$ to the nearest $\mu_k$.  
   - **Update** each $\mu_k$ to the average of points in its cluster.  
3. **Output** final clusters $C_1,\dots,C_K$ and centroids $\mu_1,\dots,\mu_K$.

---

🎯 ***Key Point***:  
The centroids are simply the **arithmetic mean** of the members of each cluster, ensuring that each step non‑increasingly optimizes the within‐cluster sum of squared distances.  

---
## 🗂 Applying K‑Means for Clustering

Given a dataset of \(N\) samples \(\{x_i\}\) with \(d\) features, here’s a step‑by‑step guide:

1. **Data Preparation**  
   - **Clean**: handle missing values (impute or remove).  
   - **Encode**: convert categorical variables (one‑hot, ordinal).  
   - **Scale**: normalize or standardize features so distances are meaningful.

2. **Choose Number of Clusters (\(K\))**  
   - **Elbow Method**: plot total within‑cluster SSE vs. \(K\) and look for the “elbow.”  
   - **Silhouette Score**: evaluate clustering quality for different \(K\).  
   - **Domain Knowledge**: leverage any prior insight on expected group count.

3. **Initialize Centroids**  
   - Randomly select \(K\) initial points, or  
   - Use **K‑Means++** for smarter seeding and faster convergence.

4. **Run the Algorithm**  
   - **Assignment**: assign each \(x_i\) to its nearest centroid \(\mu_k\).  
   - **Update**: recompute each \(\mu_k\) as the mean of its assigned points.  
   - **Repeat** until assignments stabilize or max iterations reached.

5. **Evaluate Clustering**  
   - **Inertia (SSE)**: lower is better.  
   - **Silhouette Coefficient**: closer to +1 indicates well‑separated clusters.  
   - **Visualization**: if \(d\le3\), plot clusters; otherwise use PCA/t‑SNE to project to 2D.

6. **Interpret Results**  
   - **Centroids**: inspect feature values at each \(\mu_k\) to characterize clusters.  
   - **Cluster Sizes**: check for imbalanced cluster populations.  
   - **Actionable Insights**: relate clusters back to business or scientific context.

7. **Refinement (Optional)**  
   - If clusters are unsatisfactory, consider:  
     - Changing \(K\).  
     - Trying different feature subsets.  
     - Using another clustering method (e.g., DBSCAN, Gaussian Mixtures).

---

🎯 ***Outcome***: A partition of your data into \(K\) cohesive groups, with each sample labeled by its cluster index and each cluster summarized by its centroid.  

---
## 🔢 Determining the Optimal Number of Clusters ($K$) in K‑Means

Choosing the right $K$ is crucial. Here are the most common methods:

---

### 1. Elbow Method  
- Compute the **Within‑Cluster Sum of Squares** (SSE) for $K=1,\dots,K_{\max}$:  
  $$text{SSE}(K)
  = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2.
  $$  
- Plot $\text{SSE}(K)$ vs. $K$ and look for the “elbow” point where the curve’s slope sharply decreases.

---

### 2. Silhouette Analysis  
- For each sample $i$, let  
  $$a(i) = \frac{1}{|C_k|-1}\sum_{j \in C_k,\,j\neq i} d(i,j),\quad
  b(i) = \min_{l \neq k}\frac{1}{|C_l|}\sum_{j\in C_l} d(i,j).
  $$  
- The **silhouette score** is  
  $$s(i) = \frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}},
  $$  
  ranging from $-1$ to $+1$.  
- Compute the **average** $s(i)$ over all samples for each $K$; choose the $K$ that maximizes it.

---

### 3. Gap Statistic  
- Generate $B$ reference datasets (e.g. uniform in the data’s bounding box) and compute their $\log\text{SSE}^*_b$.  
- Define  
  $$text{Gap}(K)
  = \frac{1}{B}\sum_{b=1}^B \log\text{SSE}^*_b
    \;-\;\log\text{SSE}(K).
  $$  
- Choose the smallest $K$ such that  
  $$text{Gap}(K)\;\ge\;\text{Gap}(K+1)\;-\;s_{K+1},
  $$  
  where $s_{K+1}$ is the standard deviation of the $\log\text{SSE}^*$ values.

---

### 4. Cross‑Validation & Stability  
- **Hold‑out** clustering: fit on one split, measure SSE on the hold‑out set.  
- **Resampling stability**: cluster on bootstrap samples and compare assignments; stable $K$ yields consistent clusters.

---

### 5. Domain Knowledge & Practical Constraints  
- Business or scientific context may dictate a natural number of groups.  
- Simpler models ($K$ small) are easier to interpret and maintain.

---

🎯 **Tip**: Combine two or more methods (e.g. Elbow + Silhouette) to arrive at a robust choice of $K$.  


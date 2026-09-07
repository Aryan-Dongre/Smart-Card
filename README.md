# 🛒 SmartCart – Customer Segmentation Using Unsupervised Learning

> **Discover hidden customer segments and understand purchasing behavior using unsupervised machine learning.**

SmartCart is an **unsupervised machine learning project** focused on analyzing customer characteristics and purchasing behavior to identify meaningful customer segments.

The project uses customer demographic, purchasing, engagement, and behavioral information to group similar customers without relying on predefined target labels.

The analysis implements **K-Means Clustering** and **Agglomerative Hierarchical Clustering**, with **PCA** used for dimensionality reduction and visualization.

---

## 📌 Project Overview

Businesses often have customers with very different purchasing patterns, spending levels, engagement behaviors, and demographic characteristics.

Instead of manually defining customer categories, SmartCart uses unsupervised learning to discover naturally occurring groups within the customer data.

The project follows this workflow:

```text
Raw Customer Data
       ↓
Data Exploration
       ↓
Missing Value Treatment
       ↓
Feature Engineering
       ↓
Feature Selection
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
PCA Dimensionality Reduction
       ↓
Clustering
   ↙          ↘
K-Means    Agglomerative
   ↓          ↓
Cluster Analysis
       ↓
Customer Segmentation
```

---

## 🎯 Objectives

The main objectives of the project are:

* Analyze customer demographic and purchasing information.
* Clean and preprocess the customer dataset.
* Engineer meaningful customer-level features.
* Transform categorical variables into numerical representations.
* Scale features before applying clustering algorithms.
* Reduce dimensionality using PCA.
* Determine an appropriate number of customer clusters.
* Apply K-Means clustering.
* Apply Agglomerative clustering.
* Compare and visualize customer segments.
* Characterize the resulting customer groups based on income, spending, purchasing channels, engagement, and other behavioral features.

---

## 📊 Dataset

The project uses a customer marketing dataset containing **2,240 customers and 22 original columns**.

The original dataset contains information such as:

| Category             | Features                                 |
| -------------------- | ---------------------------------------- |
| Demographics         | Year of Birth, Education, Marital Status |
| Income               | Income                                   |
| Family               | Kidhome, Teenhome                        |
| Customer Activity    | Recency                                  |
| Product Spending     | Wines, Fruits, Meat, Fish, Sweets, Gold  |
| Purchasing Channels  | Web, Catalog, Store Purchases            |
| Online Engagement    | Web Visits per Month                     |
| Customer Feedback    | Complain                                 |
| Campaign             | Response                                 |
| Customer Information | Customer Joining Date                    |

The notebook identifies **24 missing values in the `Income` column**, while the remaining original columns contain no missing values.

---

# 🔍 Exploratory Data Analysis

Initial exploration includes:

* Dataset shape
* Data types
* Missing-value analysis
* Feature inspection
* Distribution/outlier analysis
* Correlation analysis
* Visualization of customer characteristics

The dataset initially contains:

```text
Rows    : 2240
Columns : 22
```

---

# 🧹 Data Preprocessing

## 1. Missing Value Treatment

The missing values in `Income` are handled using **median imputation**.

```python
median = df["Income"].median()
df["Income"] = df["Income"].fillna(median)
```

This ensures that the clustering pipeline does not contain missing numerical values.

---

## 2. Feature Engineering

Several new features are created to better represent customer behavior.

### Age

Customer age is calculated from the birth year:

```python
df["Age"] = 2026 - df["Year_Birth"]
```

### Customer Tenure

The customer joining date is converted into a datetime, and customer tenure is calculated relative to the latest joining date in the dataset.

```python
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True
)

reference_date = df["Dt_Customer"].max()

df["Customer_Tenure_Date"] = (
    reference_date - df["Dt_Customer"]
).dt.days
```

### Total Spending

Individual product spending features are combined into one overall spending metric:

```python
df["Total_Spending"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)
```

### Total Children

The number of children and teenagers in the household is combined:

```python
df["Total_Children"] = (
    df["Kidhome"] + df["Teenhome"]
)
```

---

# 🏷️ Categorical Feature Engineering

The `Education` feature is simplified into three broader categories:

* `UnderGraduate`
* `Graduate`
* `PostGraduate`

The transformation maps:

```text
Basic / 2n Cycle → UnderGraduate
Graduation       → Graduate
Master / PhD     → PostGraduate
```

Similarly, marital-status categories are transformed into a simplified `Living_with` feature:

```text
Married / Together → Partner
Single / Divorced / Widow / Alone / Absurd / YOLO → Alone
```

---

# 🗑️ Feature Selection

Several columns are removed before clustering because they are either identifiers, redundant with engineered features, raw date fields, or individual spending components that have already been aggregated.

The removed features include:

```python
[
    "ID",
    "Year_Birth",
    "Marital_Status",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]
```

This produces a cleaned dataset with **15 features** before categorical encoding.

The retained feature space contains customer attributes such as:

```text
Education
Income
Recency
NumDealsPurchases
NumWebPurchases
NumCatalogPurchases
NumStorePurchases
NumWebVisitsMonth
Complain
Response
Age
Customer_Tenure_Date
Total_Spending
Total_Children
Living_with
```

---

# 🔢 Categorical Encoding

Categorical variables are converted into numerical representations using one-hot encoding.

The resulting encoded dataset contains features such as:

```text
Education_Graduate
Education_PostGraduate
Education_UnderGraduate
Living_with_Alone
Living_with_Partner
```

After encoding, the final feature matrix contains:

```text
2236 samples × 18 features
```

---

# ⚖️ Feature Scaling

Because clustering algorithms are sensitive to feature magnitude, the final feature matrix is standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

The resulting scaled matrix has:

```text
2236 samples × 18 features
```

---

# 📉 PCA Dimensionality Reduction

Principal Component Analysis (**PCA**) is used to reduce the 18-dimensional feature space to three principal components for visualization and clustering.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)
```

The first three principal components explain approximately:

```text
PC1 → 23.16%
PC2 → 11.39%
PC3 → 10.41%
```

Together, the three components explain approximately **44.95% of the variance** in the scaled dataset.

The reduced representation is also visualized using a 3D scatter plot.

---

# 🤖 Clustering

The project implements two unsupervised clustering approaches.

## 1. K-Means Clustering

K-Means is used to divide customers into groups based on similarity in the PCA feature space.

The project evaluates different values of `K` from 1 to 10 using the **Within-Cluster Sum of Squares (WCSS)** / inertia.

```python
wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)
```

The elbow analysis identifies:

```text
Optimal K = 4
```

The final K-Means model is therefore configured with four clusters:

```python
kmean = KMeans(
    n_clusters=4,
    random_state=42
)

labels_kmean = kmean.fit_predict(X_pca)
```

The resulting clusters are visualized in three-dimensional PCA space.

---

## 2. Silhouette Analysis

Silhouette scores are also calculated for values of `K` from 2 through 10.

```python
from sklearn.metrics import silhouette_score

ss = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = kmeans.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        labels
    )

    ss.append(score)
```

The notebook uses the silhouette score as an additional clustering-quality diagnostic.

---

## 3. Agglomerative Clustering

The project also applies **Agglomerative Hierarchical Clustering** using Ward linkage.

```python
from sklearn.cluster import AgglomerativeClustering

agg_clf = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

labels_agg = agg_clf.fit_predict(X_pca)
```

Four clusters are generated and visualized using the same three PCA dimensions.

---

# 👥 Customer Segment Analysis

The final cluster assignments from Agglomerative Clustering are added to the feature matrix:

```python
X["cluster"] = labels_agg
```

The project then analyzes the resulting segments using:

* Cluster size
* Income
* Total spending
* Purchase channels
* Web visits
* Campaign response
* Age
* Customer tenure
* Number of children
* Education
* Living arrangement

---

# 📊 Cluster Insights

The resulting four clusters show clear differences in customer behavior.

### Cluster 0 – Lower Spending Customers

Average characteristics include approximately:

* Income: **39.7K**
* Total Spending: **222**
* Web Purchases: **3.15**
* Catalog Purchases: **0.97**
* Store Purchases: **4.14**
* Total Children: **1.24**

This group represents customers with comparatively lower income and spending levels.

---

### Cluster 1 – High-Value Partner Customers

Average characteristics include approximately:

* Income: **72.8K**
* Total Spending: **1,237**
* Web Purchases: **5.69**
* Catalog Purchases: **5.50**
* Store Purchases: **8.66**
* Total Children: **0.51**
* Response: **16.7%**

This segment represents relatively high-income and high-spending customers with strong activity across multiple purchasing channels.

---

### Cluster 2 – Lower Spending Customers Living Alone

Average characteristics include approximately:

* Income: **37.0K**
* Total Spending: **166**
* Web Purchases: **2.71**
* Catalog Purchases: **0.84**
* Store Purchases: **3.62**
* Total Children: **1.27**

This group has comparatively low spending and income, with customers predominantly classified as living alone.

---

### Cluster 3 – High-Value Customers with Strong Campaign Response

Average characteristics include approximately:

* Income: **70.7K**
* Total Spending: **1,190**
* Web Purchases: **5.79**
* Catalog Purchases: **5.01**
* Store Purchases: **8.43**
* Total Children: **0.46**
* Response: **32.0%**

This segment combines high spending with the strongest campaign response among the four groups.

The cluster statistics above are directly derived from the notebook's cluster summary.

---

# 📈 Visualizations

The project includes visual analysis such as:

* Customer feature distributions
* Outlier visualization
* Correlation analysis
* Correlation heatmap
* PCA 3D projection
* K-Means cluster visualization
* Agglomerative cluster visualization
* Cluster size comparison
* Income vs. Total Spending
* Cluster-level summaries

The PCA projection is specifically used to visualize the high-dimensional customer data in three dimensions.

---

# 🧰 Tech Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Clustering

* K-Means
* Agglomerative Clustering

### Dimensionality Reduction

* PCA

### Model Selection / Evaluation

* WCSS / Elbow Method
* Silhouette Score

### Additional Utility

* `kneed` – automatic elbow detection

---

# 📂 Project Structure

```text
SmartCart/
│
├── SmartCart Project.ipynb
├── smartcart_customers.csv
└── README.md
```

> Dataset filename/path may vary depending on the local project setup.

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd SmartCart
```

## 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed jupyter
```

## 3. Start Jupyter Notebook

```bash
jupyter notebook
```

## 4. Open the notebook

```text
SmartCart Project.ipynb
```

## 5. Run all cells

Make sure the customer CSV dataset is available at the path expected by the notebook.

---

# 💡 Business Applications

The customer segments discovered through this project can potentially support:

* Customer profiling
* Personalized marketing
* Targeted promotions
* Customer retention strategies
* High-value customer identification
* Campaign targeting
* Channel-specific marketing
* Customer relationship management
* Spending behavior analysis

For example, high-spending customers can be targeted with premium offers, while lower-spending groups may benefit from introductory promotions or engagement campaigns.

---

# 🔮 Future Improvements

Potential improvements to the project include:

* Compare clustering performance on the original scaled feature space versus PCA space.
* Perform a more systematic comparison between K-Means and Agglomerative Clustering.
* Test additional clustering algorithms such as DBSCAN or Gaussian Mixture Models.
* Optimize clustering hyperparameters.
* Add interactive cluster dashboards.
* Develop automated customer segment labels.
* Build a customer segmentation API.
* Create a web dashboard for marketing teams.
* Track segment movement over time as new customer data becomes available.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

```text
Unsupervised Learning
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
PCA
        ↓
K-Means Clustering
        ↓
Agglomerative Clustering
        ↓
Elbow Method
        ↓
Silhouette Analysis
        ↓
Cluster Interpretation
```

---

# 📌 Results Summary

| Component              | Result                                               |
| ---------------------- | ---------------------------------------------------- |
| Original Dataset       | 2,240 rows × 22 columns                              |
| Missing Values         | 24 in Income                                         |
| Engineered Features    | Age, Customer Tenure, Total Spending, Total Children |
| Clean Feature Set      | 15 features before encoding                          |
| Encoded Feature Matrix | 2,236 × 18                                           |
| Scaling                | StandardScaler                                       |
| PCA Components         | 3                                                    |
| PCA Variance Explained | ~44.95%                                              |
| Optimal K from Elbow   | 4                                                    |
| K-Means Clusters       | 4                                                    |
| Agglomerative Clusters | 4                                                    |
| Clustering Type        | Unsupervised Learning                                |

---

# 👨‍💻 Author

**Aryan Dongre**

Aspiring Data Scientist / Machine Learning Developer

---

## ⭐ Project Highlights

* Real-world customer segmentation problem
* End-to-end unsupervised learning workflow
* Feature engineering for customer behavior
* PCA-based dimensionality reduction
* K-Means clustering
* Agglomerative hierarchical clustering
* Elbow method for cluster selection
* Silhouette analysis
* Business-oriented cluster interpretation

---

⭐ **If you found this project useful, consider giving the repository a star!**

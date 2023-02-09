# ----------------------
# | Preprocessing Data | 
# ---------------------- 

# Import libraries ------------------------------------------------------------
import pandas as pd
import numpy as np

# Read data -------------------------------------------------------------------
kickstarter = pd.read_excel("/Users/ylfaliang/Documents/MMA/Materials/2022 Fall/INSY662 - Data Mining and Visualization/Individual Project/Kickstarter.xlsx")

# To keep the original dataset, create "df" for further manipulation
df = kickstarter
# First na check: "category" has 1394 na values, but I decided to keep it still it is considered as a significant attricute.
df.isna().sum()

# Drop the rows with irrelevant states ----------------------------------------

# Check all states
df.state.unique()
# Drop rows with irrelevent states
df = df.drop(df[df.state == "canceled"].index)
df = df.drop(df[df.state == "suspended"].index)
# Check left states
df.state.unique()

# Transform columns -----------------------------------------------------------

# "state" to numerical values "staeResult" (created a new column and keep "state" as reference)
df["stateResult"] = np.where(df["state"] == "successful", 1, 0)

# "goal" in USD (overwrite the original column)
df["goal"] = df.apply(lambda x: x.goal * x.static_usd_rate, axis = 1)
df = df.rename(columns = {"goal":"goalUSD"})

# "popCountry" to identify if kickstarter is popular in the country or not
df["popCountry"] = df["country"].map(lambda x: 1 if (x in ["US","GB","DE"]) else 0)

# Drop insignificant columns (intuitively) ------------------------------------

# Check all columns first
df.columns
# Create a drop list
dropList = ["name", "pledged", "disable_communication", "country", "currency", "deadline", 
            "state_changed_at", "created_at", "launched_at", "backers_count", "static_usd_rate",
            "usd_pledged", "spotlight", "name_len", "blurb_len", "deadline_weekday", "state_changed_at_weekday", 
            "created_at_weekday", "launched_at_weekday", "deadline_day", "deadline_yr", 
            "deadline_hr", "state_changed_at_month", "state_changed_at_day", "state_changed_at_yr", 
            "state_changed_at_hr", "created_at_month", "created_at_day", "created_at_yr", 
            "created_at_hr", "launched_at_day", "launched_at_yr", "launched_at_hr", 
            "create_to_launch_days", "launch_to_state_change_days", "staff_pick"]

# Drop the columns that are in the drop list
df = df.drop(dropList, axis = 1)

# Drop rows with na values ----------------------------------------------------
df.isna().sum()
df = df.dropna(axis = 0, how = "any")
df.isna().sum()

# Correlation check before dummification
corrMatrix = df.loc[:, ~df.columns.isin(["id", "state", "stateResult"])].corr()

# Dummify categorical columns .................................................
df = pd.get_dummies(df, columns=["category","deadline_month","launched_at_month"])


# Feature selection after dummification ---------------------------------------
# (as there are too many columns after dummification)

X = df.loc[:, ~df.columns.isin(["id", "state", "stateResult", "goalUSD"])]
y = df[["stateResult"]]

# Standardize predictors before running lasso
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Run lasso
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.02) 
model = ls.fit(X_std,y)
model.coef_

lassoCheck = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])
dropList = lassoCheck[lassoCheck.coefficient == 0]["predictor"].tolist()

# Drop columns that are in dropList
df = df.drop(dropList, axis = 1)

# Record the remaining categories for preprocessing grading dataset
categoryList = []
for i in df.columns:
    if "category" in i:
        categoryList.append(i[9:])

# ------------------------
# | Classification Model |
# ------------------------ 

# Define the target and predictors
X = df.loc[:, ~df.columns.isin(["id", "state", "stateResult"])]
y = df[["stateResult"]]

# Standardize the predictors
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)

# Logistic Regression ---------------------------------------------------------
from sklearn.linear_model import LogisticRegression

# Model
lr = LogisticRegression(max_iter=5000, warm_start = True, multi_class = "ovr", n_jobs = 4)
model_lr = lr.fit(X_train,y_train)

# View results
model_lr.intercept_
model_lr.coef_

# Make predictions based on the test dataset
y_test_pred = model_lr.predict(X_test)

# Accuracy score
from sklearn import metrics
accuracy_lr = metrics.accuracy_score(y_test, y_test_pred)

# Confusion matrix with labels
confusion_lr = pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred, labels=[0,1]), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1'])

# Precision/Recall
precision_lr = metrics.precision_score(y_test, y_test_pred)
recall_lr = metrics.recall_score(y_test, y_test_pred)

# F1 score
f1_lr = metrics.f1_score(y_test, y_test_pred)

# KNN -------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

# Model
knn = KNeighborsClassifier(n_neighbors=20, p=1, weights = "distance")
model_knn = knn.fit(X_train,y_train)

# Make predictions based on the test dataset
y_test_pred = model_knn.predict(X_test)

# Accuracy score
accuracy_knn = metrics.accuracy_score(y_test, y_test_pred)

# Confusion matrix with labels
confusion_knn = pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred, labels=[0,1]), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1'])

# Precision/Recall
precision_knn = metrics.precision_score(y_test, y_test_pred)
recall_knn = metrics.recall_score(y_test, y_test_pred)

# F1 score
f1_knn = metrics.f1_score(y_test, y_test_pred)

# Random Forest ---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# Model
randomforest = RandomForestClassifier(n_estimators = 500, random_state=0)
model_rf = randomforest.fit(X_train, y_train)

# Make predictions based on the test dataset
y_test_pred = model_rf.predict(X_test)

# Accuracy score
accuracy_rf = metrics.accuracy_score(y_test, y_test_pred)

# Confusion matrix with labels
confusion_rf = pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred, labels=[0,1]), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1'])

# Precision/Recall
precision_rf = metrics.precision_score(y_test, y_test_pred)
recall_rf = metrics.recall_score(y_test, y_test_pred)

# F1 score
f1_rf = metrics.f1_score(y_test, y_test_pred)

# Comparison ------------------------------------------------------------------
comparison = pd.DataFrame({"Accuracy": [accuracy_lr, accuracy_knn, accuracy_rf],
                           "Precision": [precision_lr, precision_knn, precision_rf],
                           "Recall": [recall_lr, recall_knn, recall_rf],
                           "F1 Score": [f1_lr, f1_knn, f1_rf]}, index = ["Logistic Regression","KNN","Random Forest"])
from tabulate import tabulate
print(tabulate(comparison, headers = "keys", tablefmt = "fancy_grid"))

# --------------------
# | Clustering Model |
# --------------------

# Preprocessing the dataset
clusterX = kickstarter
clusterX = clusterX.drop(df[df.state == "canceled"].index)
clusterX = clusterX.drop(df[df.state == "suspended"].index)
clusterX["stateResult"] = np.where(clusterX["state"] == "successful", 1, 0)
clusterX["goal"] = clusterX.apply(lambda x: x.goal * x.static_usd_rate, axis = 1)
clusterX = clusterX.rename(columns = {"goal":"goalUSD"})
clusterX["popCountry"] = clusterX["country"].map(lambda x: 1 if (x in ["US","GB","DE"]) else 0)

dropList = ["name", "pledged", "disable_communication", "country", "currency", "deadline", 
            "state_changed_at", "created_at", "launched_at", "backers_count", "static_usd_rate",
            "usd_pledged", "spotlight", "name_len", "blurb_len", "deadline_weekday", "state_changed_at_weekday", 
            "created_at_weekday", "launched_at_weekday", "deadline_day", "deadline_yr", 
            "deadline_hr", "state_changed_at_month", "state_changed_at_day", "state_changed_at_yr", 
            "state_changed_at_hr", "created_at_month", "created_at_day", "created_at_yr", 
            "created_at_hr", "launched_at_day", "launched_at_yr", "launched_at_hr", 
            "create_to_launch_days", "launch_to_state_change_days", "deadline_month","launched_at_month","blurb_len_clean"]
clusterX = clusterX.drop(dropList, axis = 1)
clusterX = clusterX.dropna(axis = 0, how = "any")

for i in categoryList:
    clusterX["category_"+i] = clusterX["category"].map(lambda x: 1 if (i == x) else 0)
clusterX = clusterX.drop("category", axis = 1)

# Define predictors
X = clusterX.loc[:, ~clusterX.columns.isin(["id","state"])]

# Use isolation forest to find anomallies first
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=.01)

pred = pd.DataFrame(iforest.fit_predict(X), index = X.index, columns = ["pred"])
score = iforest.decision_function(X)

# Drop anomalies
anomalyIndex = pred[pred.pred == -1].index
X = pd.DataFrame(X, columns = X.columns).drop(anomalyIndex, axis = 0)

import matplotlib.pyplot as plt
plt.clf()
plt.style.use("ggplot")
plt.figure(figsize = (15, 8))
plt.hist(X.name_len_clean, bins=X.name_len_clean.nunique(), color = "royalblue", width = 0.6)
plt.xlabel("name_len_clean")
plt.ylabel("Project Count")
plt.title("name_len_clean Distribution")
plt.show()


# Standardize the predictors
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# K-Means ---------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Finding optimal K

kList = {"k":[],"silhouette":[]}
for k in range (2,9):   
    km = KMeans(n_clusters = k, random_state=0)
    model_km = km.fit(X_std)
    label = model_km.labels_
    kList["k"].append(k)
    kList["silhouette"].append(silhouette_score(X_std,label))
kList = pd.DataFrame(kList)
print(kList[kList.silhouette >= 0.25].k)

from sklearn.metrics import silhouette_samples
kTmp = []
cTmp = []
sTmp = []
for k in kList[kList.silhouette >= 0.25].k:
    km = KMeans(n_clusters = k, random_state=0)
    model_km = km.fit(X_std)
    label = model_km.labels_
    silhouette = silhouette_samples(X_std,label)
    slh = pd.DataFrame({"Label":label,"Silhouette":silhouette})
    
    for c in range(k):
        kTmp.append(k)
        cTmp.append(c)
        sTmp.append(np.average(slh[slh["Label"] == c].Silhouette))
kSummary = pd.DataFrame({"k":kTmp, "cluster":cTmp, "silhouette": sTmp}).pivot(index = "k", columns = "cluster", values = "silhouette")
print(tabulate(kSummary, headers = "keys", tablefmt = "fancy_grid"))


k = 7
km = KMeans(n_clusters = 7, random_state=0)
model_km = km.fit(X_std)
label = pd.DataFrame(model_km.predict(X_std), columns = ["kmeans"])
kmCount = pd.DataFrame(label.value_counts(), columns = ["Project_count"]).reset_index().rename(columns={"kmeans":"Group"})

kmCenter = pd.DataFrame(km.cluster_centers_, columns = X.columns)
kmCenter = round(pd.DataFrame(km.cluster_centers_, columns = X.columns), 3).reset_index().rename(columns={"index":"Group"})
clusterSummary = kmCenter.merge(kmCount, how = "left", left_on = "Group", right_on = "Group").drop("Group", axis = 1)
print(tabulate(clusterSummary.T, headers = "keys", tablefmt = "fancy_grid"))

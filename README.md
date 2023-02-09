# Kickstarter Project State Classification and Insights from Clustering
````diff 
+ (kickstarter-classification-clustering)
````
### Quick Summary
***
- **Purpose:** Perform classification techniques on Kickstarter's projects and identify the states (if projects will hit the target or not). Clustering method is presented as well to gain the insights of various projects.
- **Dataset Source:**
  - [Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects)
<br><br>

### Detailed Description
***
Preprocessing and feature selection are implemented before running the models. Logistic Regression, KNN, and Random Forest approaches are used and compared to obtain the model with the best performance. The dataset is split to 33% test data and 67% train data with random state = 5. K-Means is applied in the clustering model to help Kickstarter study whether there is any difference among project categories. The optimal number of clusters is found to be 7 with a silhouette score of 0.275. Clustering with 6 and 8 clusters also comes with ~26% overall silhouette score yet there are two clusters with low silhouette in both cases, while 7-clusters only has one.
<br><br>

### Preview
***
- Clustering Result<br>
![Clustering Result](https://user-images.githubusercontent.com/111717563/217730853-26f55686-2a5a-4d70-a8c2-81a2fffb7da5.png)

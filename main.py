#Project 1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# loading dataset
data= pd.read_csv("breast_cancer.csv")
data

# Splitting feature set and target set
x = data.iloc[:, 1:].values #featureset
y = data.iloc[:, 0].values  #targetset

label = LabelEncoder()
x[:,0] = label.fit_transform(x[:,0])
x[:,1] = label.fit_transform(x[:,1])
x[:,2] = label.fit_transform(x[:,2])
x[:,3] = label.fit_transform(x[:,3])
x[:,4] = label.fit_transform(x[:,4])
x[:,6] = label.fit_transform(x[:,6])
x[:,7] = label.fit_transform(x[:,7])
x[:,8] = label.fit_transform(x[:,8])

# splitting into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# scale features
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#feature selection
logistic = LogisticRegression()
selector = RFE(logistic, n_features_to_select=5, step=1)
selector = selector.fit(x_train, y_train)
x_train= selector.transform(x_train)
x_test = selector.transform(x_test)

#Logistic Regression
param_grid_Lr = {'C': [0.01, 0.1, 1, 10, 100]}
Lr = LogisticRegression(random_state=0)
grid_search_Lr = GridSearchCV(Lr, param_grid_Lr, cv=5)
grid_search_Lr.fit(x_train, y_train)
Lr_best = grid_search_Lr.best_estimator_

#K Nearest Neighbors(KNN)
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)
grid_search_knn.fit(x_train, y_train)
knn_best = grid_search_knn.best_estimator_

#Naive Bayes model
nb = GaussianNB()
nb.fit(x_train, y_train)

# prediction on test set
y_pred_Lr = Lr_best.predict(x_test)
y_pred_knn = knn_best.predict(x_test)
y_pred_nb = nb.predict(x_test)

# Performance evaluation
acc_Lr = accuracy_score(y_test, y_pred_Lr)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_nb = accuracy_score(y_test, y_pred_nb)

cm_Lr = confusion_matrix(y_test, y_pred_Lr)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_nb = confusion_matrix(y_test, y_pred_nb)

cr_Lr = classification_report(y_test, y_pred_Lr)
cr_knn = classification_report(y_test, y_pred_knn)
cr_nb = classification_report(y_test, y_pred_nb)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Naive Bayes'],
    'Accuracy': [acc_Lr, acc_knn, acc_nb],
    'Confusion Matrix': [cm_Lr, cm_knn, cm_nb],
    'Classification Report': [cr_Lr, cr_knn, cr_nb]
})

results.to_csv('results.csv', index=False)
results = pd.read_csv('results.csv')
print(results)



#Project 2
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
url = "https://news.abplive.com/news/india/india-s-approach-not-limited-to-health-but-also-wellness-pm-modi-addresses-post-budget-session-1586419"
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')

text = ""
for p in soup.find_all('p'):
    text += p.get_text()
nltk.download('stopwords')

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if not word in stop_words]
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

token = []
for word in filtered_tokens:
    if sia.polarity_scores(word)['compound'] > 0:
        token.append((word, 'positive'))
    elif sia.polarity_scores(word)['compound'] < 0:
        token.append((word, 'negative'))
    else:
        token.append((word, 'neutral'))

train_size = int(0.8 * len(token))
train_data = token[:train_size]
test_data  = token[train_size:]

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform([t[0] for t in train_data])
train_labels = [t[1] for t in train_data]

test_vectors = vectorizer.transform([t[0] for t in test_data])
test_labels = [t[1] for t in test_data]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_vectors, train_labels)
predicted_labels = knn.predict(test_vectors)

print(classification_report(test_labels, predicted_labels))
print(confusion_matrix(test_labels, predicted_labels))
text_vector = vectorizer.transform([word for word in filtered_tokens])
predicted_sentiment = knn.predict(text_vector)

positive_count=sum(1 for label in predicted_sentiment if label == 'positive')
negative_count=sum(1 for label in predicted_sentiment if label == 'negative')
neutral_count=sum(1 for label in predicted_sentiment if label == 'neutral')
sentiment_score=(positive_count - negative_count) / len(predicted_sentiment)
print("Sentiment score:", sentiment_score)
print("Positive count:", positive_count)
print("Negative count:", negative_count)
print("Neutral count:", neutral_count)

if sentiment_score > 0:
    print("The sentiment of the text is positive.")
elif sentiment_score < 0:
    print("The sentiment of the text is negative.")
else:
    print("The sentiment of the text is neutral.")



#Project 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('CC GENERAL.csv')
data.dtypes
data.describe()

data=data.drop('CUST_ID',axis=1)

#Checking the correlation between variables
sns.heatmap(data.corr())
plt.show()

# Checking for missing values
print(data.isnull().sum())

# Fill missing values with the mean of the column
data.fillna(data.mean(), inplace=True)
print(data.isnull().sum())

clustering_data= data[['CREDIT_LIMIT', 'PURCHASES', 'PURCHASES_FREQUENCY', 'PAYMENTS', 'TENURE','BALANCE']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

wcss = []
for i in range(1, 11):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Kmeans clustering

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(scaled_data)
pred_y = kmeans.fit_predict(scaled_data)

# adding cluster labels to the original dataset
data['cluster'] = kmeans.labels_

sns.scatterplot(data=data, x='PURCHASES', y='CASH_ADVANCE', hue='cluster', palette='bright')
plt.title('Clustered Credit Card Users')
plt.show()


print(data.groupby('cluster').mean())
#hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
hc.fit(scaled_data)
pred_y = hc.fit_predict(scaled_data)
# adding cluster labels to the original dataset
data['cluster'] = hc.labels_


sns.scatterplot(data=data, x='PURCHASES', y='CASH_ADVANCE', hue='cluster', palette='bright')
plt.title('Clustered Credit Card Users')
plt.show()

print(data.groupby('cluster').mean())


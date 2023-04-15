import pandas as pd
import numpy as np
import nltk, re, spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer, ColumnTransformer
from nltk import pos_tag, ne_chunk
from nltk import WordNetLemmatizer, Tree
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from spacy.lang.en.stop_words import STOP_WORDS

df = pd.read_csv('train.csv')
df.head()
df.rating.isna().sum()
df.rating.value_counts() # positive(rating>7)--1, negative(rating<=7)--0
df.rating = np.where(df.rating > 7, 1, 0) 

X = df.loc[:, ('benefits_review', 'side_effects_review', 'comments_review')]
y = df.rating

#%% Adding features into data preprocessing
# replace special characters (preserve space)
X = X.apply(lambda x: [re.sub('[^a-z0-9]', ' ', i) for i in x])

# tokenize columns
X = X.apply(lambda x: [word_tokenize(i) for i in x])

# remove stop words
X = X.apply(
  lambda x: [
  [w for w in tokenlist if w not in stopwords.words('english')]
  for tokenlist in x
  ]
)

# Part-of-Speech Tagging
X = X.apply(lambda x: [pos_tag(i) for i in x])

# Named Entity
X = X.apply(lambda x: [ne_chunk(i) for i in x])

# Lemmatization
def lemmatize_word(word):
    if isinstance(word, Tree):
        return word
    else:
        return WordNetLemmatizer().lemmatize(word)
X = X.apply(lambda x: [lemmatize_word(i) for i in x])
X = X.astype(str)

# Using the most important 1000 features in each column
#%% Using all columns
# put all words and its frequency in one list
X0 = X['benefits_review'].str.cat(X['side_effects_review'], sep=' ').str.cat( X['comments_review'], sep = ' ')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X0, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## KNN
knn = KNeighborsClassifier()
knn.fit(X_train_features, y_train)
y_pred = knn.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Naive Bayes
nb = GaussianNB()
nb.fit(X_train_features.toarray(), y_train)
y_pred = nb.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Extra Tree
et = ExtraTreesClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using benefits_review
# put all words and its frequency in one list
X1 = X['benefits_review']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using side_effects_review
# put all words and its frequency in one list
X2 = X['side_effects_review']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using comments_review
# put all words and its frequency in one list
X3 = X['comments_review']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using benefits_review and side_effects_review
# put all words and its frequency in one list
X12 = X['benefits_review'].str.cat(X['side_effects_review'], sep = ' ')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X12, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using benefits_review and comments_review
# put all words and its frequency in one list
X13 = X['benefits_review'].str.cat(X['comments_review'], sep = ' ')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X13, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using side_effects_review and comments_review
# put all words and its frequency in one list
X23 = X['side_effects_review'].str.cat(X['comments_review'], sep = ' ')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X23, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2), max_features = 1000)

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

# Using all features
#%% Using all columns
# put all words and its frequency in one list
X_0 = X['benefits_review'].str.cat(X['side_effects_review'], sep = ' ').str.cat(X['comments_review'], sep = ' ')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_0, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2))

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using benefits_review
# put all words and its frequency in one list
X_1 = X['benefits_review']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2))

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using side_effects_review
# put all words and its frequency in one list
X_2 = X['side_effects_review']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2))

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

#%% Using side_effects_review and comments_review
# put all words and its frequency in one list
X_23 = X['side_effects_review'].str.cat(X['comments_review'], sep = ' ')

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_23, y, test_size=0.3, stratify=y)

# Tfidf Vectorization
vect = TfidfVectorizer(ngram_range = (1, 2))

# train - vectorizer
X_train_features = vect.fit_transform(X_train)
features_names = vect.get_feature_names_out()
X_train_features.toarray()
# test - vectorizer
X_test_docs = [doc for doc in X_test]
X_test_features = vect.transform(X_test_docs)
X_test_features.toarray()

# Model
## SVM
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(X_train_features, y_train)
y_pred = lin_svc.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Logistic Regression
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(X_train_features, y_train)
y_pred = lr.predict(X_test_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(X_train_features, y_train)
y_pred = rf.predict(X_test_features.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score(y_test, y_pred)

# Conclusion: Using all columns, side_effects_review + comments_review columns, side_effects_review column
# can have a better accuracy and F1 score when predicting in training dataset


#%% Testing in test dataset
test = pd.read_csv('test.csv')
test.rating = np.where(test.rating > 7, 1, 0)
test_X = test.loc[:, ('benefits_review', 'side_effects_review', 'comments_review')]
test_y = test.rating
test_X = test_X.apply(lambda x: [re.sub('[^a-z0-9]', ' ', i) for i in x])
test_X = test_X.apply(lambda x: [word_tokenize(i) for i in x])
test_X = test_X.apply(
    lambda x: [
    [w for w in tokenlist if w not in stopwords.words('english')]
    for tokenlist in x
    ]
)
test_X = test_X.apply(lambda x: [pos_tag(i) for i in x])
test_X = test_X.apply(lambda x: [ne_chunk(i) for i in x])
def lemmatize_word(word):
    if isinstance(word, Tree):
        return word
    else:
        return WordNetLemmatizer().lemmatize(word)
test_X = test_X.apply(lambda x: [lemmatize_word(i) for i in x])
test_X = test_X.astype(str)

#%% Using all columns
test_X0 = test_X['benefits_review'].str.cat(test_X['side_effects_review'], sep = ' ').str.cat(test_X['comments_review'], sep = ' ')
vect = TfidfVectorizer(ngram_range = (1, 2))
test_X_features = vect.fit_transform(test_X0)
features_names = vect.get_feature_names_out()
test_X_features.toarray()
## Logistic Regression 
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(test_X_features, test_y)
y_pred = lr.predict(test_X_features)
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred)
## SVM 
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(test_X_features, test_y)
y_pred = lin_svc.predict(test_X_features)
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred) 
## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(test_X_features, test_y)
y_pred = rf.predict(test_X_features.toarray())
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred)

#%% Using side_effects_review and comments_review columns
test_X23 = test_X['side_effects_review'].str.cat(test_X['comments_review'], sep = ' ')
vect = TfidfVectorizer(ngram_range = (1, 2))
test_X_features = vect.fit_transform(test_X23)
features_names = vect.get_feature_names_out()
test_X_features.toarray()
## Logistic Regression 
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(test_X_features, test_y)
y_pred = lr.predict(test_X_features)
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred)
## SVM 
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(test_X_features, test_y)
y_pred = lin_svc.predict(test_X_features)
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred) 
## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(test_X_features, test_y)
y_pred = rf.predict(test_X_features.toarray())
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred)

#%% Using side_effects_review column
test_X2 = test_X['side_effects_review']
vect = TfidfVectorizer(ngram_range = (1, 2))
test_X_features = vect.fit_transform(test_X2)
features_names = vect.get_feature_names_out()
test_X_features.toarray()
## Logistic Regression 
lr = LogisticRegression(max_iter = 12000, random_state = 1234)
lr.fit(test_X_features, test_y)
y_pred = lr.predict(test_X_features)
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred)
## SVM 
lin_svc = LinearSVC(max_iter = 12000, random_state = 1234)
lin_svc.fit(test_X_features, test_y)
y_pred = lin_svc.predict(test_X_features)
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred) 
## Random Forest
rf = RandomForestClassifier(random_state = 1234)
rf.fit(test_X_features, test_y)
y_pred = rf.predict(test_X_features.toarray())
accuracy = metrics.accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
f1_score(test_y, y_pred)

#%% 
# Train dataset
train = pd.read_csv('train.csv')
train.rating = np.where(train.rating > 7, 1, 0) 
train_X = train.loc[:, ('benefits_review', 'side_effects_review', 'comments_review')]
train_y = train.rating
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, stratify=train_y)
X_train_docs = [doc for doc in X_train.side_effects_review]
pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=(1,2), 
                                    stop_words='english')),
            ('cls', LinearSVC())
])
pipeline.fit(X_train_docs, y_train)
training_accuracy = cross_val_score(pipeline, X_train_docs, y_train, cv=5).mean()
print("Training accuracy:", training_accuracy)
predicted = pipeline.predict([doc for doc in X_test.side_effects_review])
validation_accuracy = metrics.accuracy_score(y_test, predicted)
f1_score = f1_score(y_test, predicted)
print("Validation accuracy:", validation_accuracy)
print("F1 score:", f1_score)

# Test dataset
test = pd.read_csv('test.csv')
test.rating = np.where(test.rating > 7, 1, 0) 
test_X = test.loc[:, ('benefits_review', 'side_effects_review', 'comments_review')]
test_y = test.rating
X_docs = [doc for doc in test_X.side_effects_review]
pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=(1,2), 
                                    stop_words='english')),
            ('cls', LinearSVC())
])
pipeline.fit(X_docs, test_y)
predicted = pipeline.predict(X_docs)
test_accuracy = metrics.accuracy_score(test_y, predicted)
print("Test accuracy:", test_accuracy)
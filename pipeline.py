import pandas as pd
import numpy as np
import nltk, re, spacy, pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.linear_model import LogisticRegression
from nltk import pos_tag, ne_chunk
from nltk import WordNetLemmatizer, Tree
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import f1_score



df = pd.read_csv('train.csv')
df.rating = np.where(df.rating > 7, 'positive', 'negative') 

X = df.loc[:, ('benefits_review', 'side_effects_review', 'comments_review')]
y = df.rating


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

class TextPreprocessor(TransformerMixin, BaseEstimator):
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.apply(lambda x: ' '.join(x), axis=1)
        X = X.apply(lambda x: re.sub('[^a-z0-9]', ' ', x))
        X = X.apply(lambda x: word_tokenize(x))
        X = X.apply(lambda x: [w for w in x if w not in stopwords.words('english')])
        X = X.apply(lambda x: pos_tag(x))
        X = X.apply(lambda x: ne_chunk(x))
        X = X.apply(lambda x: [self.lemmatizer.lemmatize(w) if not isinstance(w, Tree) else w for w, t in x])
        X = X.apply(lambda x: ' '.join(x))
        return X


pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vect', TfidfVectorizer(ngram_range = (1, 2))),
    ('cls', LinearSVC(max_iter = 12000))
])

pipeline.fit(X_train, y_train)
training_accuracy = cross_val_score(pipeline, X_train, y_train, cv=5).mean()
predicted = pipeline.predict(X_test)
validation_accuracy = metrics.accuracy_score(y_test, predicted)
score = f1_score(y_test, predicted, pos_label='positive')

print("Training Accuracy: ", training_accuracy)
print("Validation Accuracy: ", validation_accuracy)
print("F1 score: ", score)
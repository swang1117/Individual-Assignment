import pandas as pd
import numpy as np
import nltk, re, spacy, pickle, os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk import pos_tag, ne_chunk
from nltk import WordNetLemmatizer, Tree
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import streamlit as st
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from streamlit_echarts import st_echarts
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image, ImageOps
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

st.set_page_config(page_title="Rating Analyzer", layout="wide")
st.title("Rating Analyzer")
data_into, model_sample, model_perf, model_test = st.tabs(["Data Introduction", "Model Sample", "Model Performance", "Model Testing"])

with data_into:

    st.markdown("""
    <style>
    .custom-title {
        font-size: 24px;
        font-weight: bold;
        color: grey;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('''
    <span class="custom-title">**Data contains four columns:**</span> 
    - benefits_review
    - side_effects_review
    - comments_review
    - rating

    ---

    <span class="custom-title">**About "rating" column:**</span>  
    - If rating score >= 7, it will be converted "Positive" 
    - If rating score < 7, it will be converted "Negative" 

    ---

    <span class="custom-title">**Data Processing steps:**</span> 
    - Remove white space
    - Tokenization : break unstructured data text into information that can be considered as discrete elements
    - Remove stop words
    - POS tagging: describe the grammatical function of a word
    - Named Entity: involves the identification of key information in the text and classification into categories
    - Lemmatization: describes the process of grouping together the different inflected forms of a word
    - TF-IDF: a numerical statistic that is intended to reflect how important a word is to a document

    ---

    <span class="custom-title">**About "benefits_review, side_effects_review, comments_review" columns:**</span> 
    - After comparing Accuracy and F1 score, using all columns and SVM model to predict rating score gets the best performance.
    ''', unsafe_allow_html=True)


with model_sample:
    test = pd.read_csv('test.csv')
    test.rating = np.where(test.rating > 7, 'positive', 'negative')

    st.subheader("Sample Posts")
    if st.button("Show Sample Posts and Sentiment"):
        placeholder = st.empty()
        with placeholder.container():
            for index, row in test.sample(10).iterrows():
                text = (row["benefits_review"] + " " + row["side_effects_review"] + " " + row["comments_review"]).strip()
                if text != "":
                    col1, col2 = st.columns([3,1])
                    with col1:
                        with st.expander(text[:100] + "..."):
                            st.write(text)
                    with col2:
                        if row["rating"] == "positive":
                            st.info(row['rating'])
                        else:
                            st.error(row['rating'])
        if st.button("Clear", type="primary"):
            placeholder.empty()

    "---"
    st.subheader("Histogram of Ratings")
    rating_value_counts = test['rating'].value_counts().reset_index()
    rating_value_counts.columns = ['rating', 'count']  # Rename columns for clarity
    colors = px.colors.qualitative.Plotly
    fig = px.bar(
        rating_value_counts,
        x='rating',  # Correct column name
        y='count',   # Correct column name
        labels={'rating': 'Rating', 'count': 'Frequency'},
        color_discrete_sequence=colors
    )
    fig.update_traces(texttemplate='%{y}', textposition='outside', textfont={'size': 12})
    st.plotly_chart(fig)

    "---"
    st.subheader("Word Cloud")

    # Replace NaN values with an empty string and ensure all values are strings
    test_pos = test.loc[test.rating == "positive", ["benefits_review", "side_effects_review", "comments_review"]].fillna("")
    test_pos = test_pos.applymap(str)  # Convert all values to strings
    test_pos_text = ' '.join(test_pos["benefits_review"].tolist() + test_pos["side_effects_review"].tolist() + test_pos["comments_review"].tolist())
    
    test_neg = test.loc[test.rating == "negative", ["benefits_review", "side_effects_review", "comments_review"]].fillna("")
    test_neg = test_neg.applymap(str)  # Convert all values to strings
    test_neg_text = ' '.join(test_neg["benefits_review"].tolist() + test_neg["side_effects_review"].tolist() + test_neg["comments_review"].tolist())
    
    # Generate word clouds
    stop_words_list = nltk.corpus.stopwords.words('english')
    col1, col2 = st.columns(2)
    
    with col1:
        wc1 = WordCloud(background_color=None, mode="RGBA", max_words=2000, stopwords=set(stop_words_list))
        wc1.generate(test_pos_text)
        st.image(wc1.to_array(), caption="Positive Reviews Word Cloud")
    
    with col2:
        wc2 = WordCloud(background_color=None, mode="RGBA", max_words=2000, stopwords=set(stop_words_list))
        wc2.generate(test_neg_text)
        st.image(wc2.to_array(), caption="Negative Reviews Word Cloud")


accuracy = None
test_f1_score = None

with model_perf:
    class TextPreprocessor(TransformerMixin, BaseEstimator):
        
        def __init__(self):
            self.lemmatizer = WordNetLemmatizer()
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            if isinstance(X, list):
                X = pd.Series(X)

            X = X.apply(lambda x: re.sub('[^a-z0-9]', ' ', x))
            X = X.apply(lambda x: word_tokenize(x))
            X = X.apply(lambda x: [w for w in x if w not in stopwords.words('english')])
            X = X.apply(lambda x: pos_tag(x))
            X = X.apply(lambda x: ne_chunk(x))
            X = X.apply(lambda x: [self.lemmatizer.lemmatize(w) if not isinstance(w, Tree) else w for w, t in x])
            X = X.apply(lambda x: ' '.join(x))
            return X

    if st.button('Build My Machine Learning Model'):
        # Replace NaN values with an empty string and ensure all values are strings
        test_X = test.loc[:, ['benefits_review', 'side_effects_review', 'comments_review']].fillna("")
        test_X = test_X.applymap(str)  # Convert all values to strings
        test_X = test_X.apply(lambda x: ' '.join(x), axis=1)  # Concatenate text from columns

        test_y = test.rating
        pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('vect', TfidfVectorizer(ngram_range = (1, 2))),
        ('cls', LinearSVC(max_iter = 12000))
    ])
        pipeline.fit(test_X, test_y)
        y_pred = pipeline.predict(test_X)
        accuracy = metrics.accuracy_score(test_y, y_pred)
        test_f1_score = metrics.f1_score(test_y, y_pred, pos_label='positive')
        
        with open('pipeline.pkl', 'wb') as f:
            pickle.dump(pipeline, f) 

    st.subheader('Model Performence')

    if accuracy and f1_score:
        st.write("Test accuracy", accuracy)
        st.write("F1 score ", test_f1_score)
    else:
        st.write("Please build the model first.")

with model_test:
    st.subheader("Testing the Model")

    review_text = st.text_area("rating test")
    review_series = pd.Series([review_text])

    if st.button("Predict"):
        if os.path.exists('pipeline.pkl') and os.path.getsize('pipeline.pkl') > 0:
            with open('pipeline.pkl', 'rb') as f:
                pipeline = pickle.load(f)
            rating = pipeline.predict(review_series)
            st.write('Predicted rating is:', rating[0])
        else:
            st.warning("Please build the model first by clicking 'Build My Machine Learning Model' button in the 'Model Performance' tab.")

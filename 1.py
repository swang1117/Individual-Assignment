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

X_processed = pipeline.fit(test_X, test_y)
predicted = pipeline.predict(test_X)
validation_accuracy = metrics.accuracy_score(test_y, predicted)
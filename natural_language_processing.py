# Natural Language Processing tutorial from Machine Learning A-Z - SuperDataScience -> Input by Ryan L Buchanan 16OCT20

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Clean the texts
import re 
# NLTK removes the stop words 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# Stemming takes the root of a word removing conjugation to simplify & understand gist meaning (reducing final dimension )
from nltk.stem.porter import PorterStemmer
# Corpus will contain all the different restaurant reviews cleaned
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
print(corpus)

# Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Tokenization
len(X[0])

# Split the dataset into the Training set and the Test set


# Train the Naive Bayes model on the Training set


# Predict the Test set results


# Make the Confusion Matrix

 

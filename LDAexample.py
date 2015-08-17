from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

from variationalLDA import LDA

categories = ['soc.religion.christian', 'comp.graphics', 'talk.politics.guns']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
count_vect = CountVectorizer(stop_words="english", min_df=5, max_df=400, max_features=1000)
X_train_counts = count_vect.fit_transform(twenty_train.data)

alpha, eta, iteration, seed = 0.3, 0.3, 4, 20
K = len(categories)

model = LDA(K, eta, [alpha] * K, X_train_counts, count_vect, seed)
model.run(iteration)

pprint(model.getBetaTopN(10))

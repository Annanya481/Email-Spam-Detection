#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

df = pd.read_csv('../datasets/mail_data.csv')
df['Category'] = df.Category.map({'ham': 1, 'spam':0})

#count_vector = CountVectorizer()
vectorizer = pickle.load(open("Vectorizer.pickle", "rb"))

training_data = vectorizer.fit_transform(df['Message'])

#naive_bayes = MultinomialNB()
model = RandomForestClassifier(n_estimators=100, criterion='gini')
model.fit(training_data, df['Category'])

pickle.dump(model, open('RandomForestModel.joblib','wb'))
#pickle.dump(vectorizer.vocabulary_, open("training_vocab.pkl", 'wb'))

email = ["Nah I don't think he goes to usf, he lives around here though"]
#test = CountVectorizer(vocabulary=count_vector.vocabulary_)
email = vectorizer.transform(email)

print(model.predict(email))
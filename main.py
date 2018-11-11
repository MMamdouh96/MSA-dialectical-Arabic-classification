import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score

# Read Data and split it
df = pd.read_csv("/home/mmamdouh/PycharmProjects/ML_Task/dataset.tsv", delimiter="\t")
pd.set_option('max_colwidth', 250)

NOL = (df['label'].value_counts()).count()

train, test = train_test_split(df, test_size=0.2, train_size=0.8, random_state=42)

# Feature Extraction with TFIDF
vectorizer = TfidfVectorizer()
vectorizerArr = vectorizer.fit_transform(train['text']) #fit_transform to extract the vocab and transform to TFIDF vector
testVector = vectorizer.transform(test['text']) #transform only to transform to TFIDF vector

# LogisticRegression Model
LogisticRegression_clf = LogisticRegression(solver='liblinear', multi_class='ovr') #liblineaer because it's good for small dataset and OVR for binary classification
LogisticRegression_clf = LogisticRegression_clf.fit(vectorizerArr, train['label'])
LogisticRegression_prediction = LogisticRegression_clf.predict(testVector)
LogisticRegression_accuracy = LogisticRegression_clf.score(testVector, test['label'])

# Calculate the average of F measure
LogisticRegression_arF = f1_score(test['label'], LogisticRegression_prediction, average='binary', pos_label='ar')
LogisticRegression_arzF = f1_score(test['label'], LogisticRegression_prediction, average='binary', pos_label='arz')
LogisticRegression_finalF = ((LogisticRegression_arF + LogisticRegression_arzF) / NOL)
print('Accuracy for LogisticRegression Model is => ' + str(LogisticRegression_accuracy * 100) + '% and F measure is => ' + str(LogisticRegression_finalF * 100) + '%')

# LinearSVC Model
svc_clf = LinearSVC()
svc_clf = svc_clf.fit(vectorizerArr, train['label'])
svc_prediction = svc_clf.predict(testVector)
svc_accuracy = svc_clf.score(testVector, test['label'])

# Calculate the average of F measure
svc_ar = f1_score(test['label'], svc_prediction, average='binary', pos_label='ar')
svc_arF = f1_score(test['label'], svc_prediction, average='binary', pos_label='arz')
svc_finalF = ((svc_arF + svc_ar) / NOL)
print('Accuracy for LinearSVC Model is => ' + str(svc_accuracy * 100) + '% and F measure is => ' + str(svc_finalF * 100) + '%')
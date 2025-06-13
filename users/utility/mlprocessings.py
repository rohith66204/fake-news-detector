import os

# import matplotlib.pyplot as plt
import pandas as pd
from django.conf import settings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

path1 = os.path.join(settings.MEDIA_ROOT, 'FinalDataSet.csv')
df = pd.read_csv(path1, encoding="ISO-8859-1")
df.shape
# plt.figure(figsize=(7, 7))
sorted_counts = df['FinalLabel'].value_counts()
df['FinalLabel'] = df.FinalLabel.map({'REAL': 1, 'FAKE': 0})
X_train, X_test, y_train, y_test = train_test_split(df['text'],
                                                    df['FinalLabel'],
                                                    random_state=42)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
# Instantiate the CountVectorizer method
count_vector = CountVectorizer(stop_words='english', lowercase=True)

training_data = count_vector.fit_transform(X_train)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


def start_logisticRegression():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(training_data, y_train)
    import pickle
    file = 'fakenews.alex'
    pickle.dump(model, open(file,'wb'))
    y_pred = model.predict(testing_data)
    cr_lg = classification_report(y_pred, y_test, output_dict=True)
    cr_lg = pd.DataFrame(cr_lg).transpose()
    cr_lg = pd.DataFrame(cr_lg)
    cr_lg = cr_lg.to_html
    return cr_lg

def start_naivebayes():
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(training_data.toarray(), y_train)
    y_pred = model.predict(testing_data.toarray())
    cr_nb = classification_report(y_pred, y_test, output_dict=True)
    cr_nb = pd.DataFrame(cr_nb).transpose()
    cr_nb = pd.DataFrame(cr_nb)
    cr_nb = cr_nb.to_html
    return cr_nb


def start_svm():
    from sklearn.svm import SVC
    model = SVC()
    model.fit(training_data, y_train)
    y_pred = model.predict(testing_data)
    cr_svm = classification_report(y_pred, y_test, output_dict=True)
    cr_svm = pd.DataFrame(cr_svm).transpose()
    cr_svm = pd.DataFrame(cr_svm)
    cr_svm = cr_svm.to_html
    return cr_svm

def start_recurrentNeurals():
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 10), max_iter=200, verbose=True)
    model.fit(training_data, y_train)
    y_pred = model.predict(testing_data)
    cr_rnn = classification_report(y_pred, y_test, output_dict=True)
    cr_rnn = pd.DataFrame(cr_rnn).transpose()
    cr_rnn = pd.DataFrame(cr_rnn)
    cr_rnn = cr_rnn.to_html
    return cr_rnn
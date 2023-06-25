# LIBRARIES
import pandas as pd
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import warnings
warnings.simplefilter("ignore")

# LOADING DATASETS
df = pd.read_csv("Dataset\\cleaned_data.csv", encoding="latin1") #due to special charas should be encoded as latin 1

toCheck = pd.read_csv("Dataset\\updated_data.csv", encoding="latin1")

# REMOVE MAX
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# DROP EXTRA COLUMNS
df.drop(['Unnamed: 0'], axis=1, inplace=True)
toCheck.drop(['Unnamed: 0'], axis=1, inplace=True)

# CHECKING WHICH ROW IS NULL FROM PRE-PROCESSING
checkNULL = df.isnull()
checkNULL = checkNULL.any(axis=1)
df[checkNULL]

toCheck = toCheck.drop_duplicates().reset_index(drop=True)

toCheck.iloc[[191, 523, 1072, 1111, 1230, 1316], [3, 4]]

# DROP THE NULL ROWS
df = df.dropna(how='any', axis=0)

# UPDATED VP VALUES 
df["verified_purchase"].value_counts(normalize=True)

# ASSIGN THE VARIABLES
X = df['review_text'] #input var
y = df['verified_purchase'] #target var

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    df['review_text'], df['verified_purchase'], test_size=0.4, random_state=42) #40% gives best results, 42 is no of life...

entiredf = format(df.shape[0])
traindf = format(X_train.shape[0])
testdf = format(X_test.shape[0])

print('Number of rows:')
print('Entire dataset:', entiredf)
print('Train dataset:', traindf)
print('Test dataset:', testdf)

# COUNT VECTORIZER AND MODELING
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer.fit(X_train)

train_c = count_vectorizer.fit_transform(X_train)
test_c = count_vectorizer.transform(X_test)

# Multinomial Naive Bayes model
mnb1 = MultinomialNB()
mnb1.fit(train_c, y_train)
prediction = mnb1.predict(test_c)

# EVALUATION
mnb_a1 = accuracy_score(y_test, prediction) * 100
mnb_p1 = precision_score(y_test, prediction) * 100
mnb_r1 = recall_score(y_test, prediction) * 100
mnb_f11 = f1_score(y_test, prediction) * 100

# Print Confusion Matrix
cm1 = confusion_matrix(y_test, prediction)
cmd1 = ConfusionMatrixDisplay(cm1, display_labels=mnb1.classes_)
cmd1.plot()

print("Multinomial Naive Bayes (Count Vectorizer):")
print("Accuracy:", mnb_a1)
print("Precision:", mnb_p1)
print("Recall:", mnb_r1)
print("F1-Score:", mnb_f11)
print()

# Support Vector Machine model
svm1 = SVC(kernel='linear')
svm1.fit(train_c, y_train)
prediction = svm1.predict(test_c)

# EVALUATION
svm_a1 = accuracy_score(y_test, prediction) * 100
svm_p1 = precision_score(y_test, prediction) * 100
svm_r1 = recall_score(y_test, prediction) * 100
svm_f11 = f1_score(y_test, prediction) * 100

# Print Confusion Matrix
cm2 = confusion_matrix(y_test, prediction)
cmd2 = ConfusionMatrixDisplay(cm2, display_labels=svm1.classes_)
cmd2.plot()

print("Support Vector Machine (Count Vectorizer):")
print("Accuracy:", svm_a1)
print("Precision:", svm_p1)
print("Recall:", svm_r1)
print("F1-Score:", svm_f11)
print()

# Logistic Regression model
lr1 = LogisticRegression()
lr1.fit(train_c, y_train)
prediction = lr1.predict(test_c)

# EVALUATION
lr_a1 = accuracy_score(y_test, prediction) * 100
lr_p1 = precision_score(y_test, prediction) * 100
lr_r1 = recall_score(y_test, prediction) * 100
lr_f11 = f1_score(y_test, prediction) * 100

# Print Confusion Matrix
cm3 = confusion_matrix(y_test, prediction)
cmd3 = ConfusionMatrixDisplay(cm3, display_labels=lr1.classes_)
cmd3.plot()

print("Logistic Regression (Count Vectorizer):")
print("Accuracy:", lr_a1)
print("Precision:", lr_p1)
print("Recall:", lr_r1)
print("F1-Score:", lr_f11)
print()

# TFIDF VECTORIZER AND MODELING
TFIDF_vectorizer = TfidfVectorizer(stop_words='english')

TFIDF_vectorizer.fit(X_train)

train_tf = TFIDF_vectorizer.fit_transform(X_train)
test_tf = TFIDF_vectorizer.transform(X_test)

# Multinomial Naive Bayes model
mnb2 = MultinomialNB()
mnb2.fit(train_tf, y_train)
prediction = mnb2.predict(test_tf)

# EVALUATION
mnb_a2 = accuracy_score(y_test, prediction) * 100
mnb_p2 = precision_score(y_test, prediction) * 100
mnb_r2 = recall_score(y_test, prediction) * 100
mnb_f12 = f1_score(y_test, prediction) * 100

# Print Confusion Matrix
cm4 = confusion_matrix(y_test, prediction)
cmd4 = ConfusionMatrixDisplay(cm4, display_labels=mnb2.classes_)
cmd4.plot()

print("Multinomial Naive Bayes (TFIDF Vectorizer):")
print("Accuracy:", mnb_a2)
print("Precision:", mnb_p2)
print("Recall:", mnb_r2)
print("F1-Score:", mnb_f12)
print()

# Support Vector Machine model
svm2 = SVC(kernel='linear')
svm2.fit(train_tf, y_train)
prediction = svm2.predict(test_tf)

# EVALUATION
svm_a2 = accuracy_score(y_test, prediction) * 100
svm_p2 = precision_score(y_test, prediction) * 100
svm_r2 = recall_score(y_test, prediction) * 100
svm_f12 = f1_score(y_test, prediction) * 100

# Print Confusion Matrix
cm5 = confusion_matrix(y_test, prediction)
cmd5 = ConfusionMatrixDisplay(cm5, display_labels=svm2.classes_)
cmd5.plot()

print("Support Vector Machine (TFIDF Vectorizer):")
print("Accuracy:", svm_a2)
print("Precision:", svm_p2)
print("Recall:", svm_r2)
print("F1-Score:", svm_f12)
print()

# Logistic Regression model
lr2 = LogisticRegression()
lr2.fit(train_tf, y_train)
prediction = lr2.predict(test_tf)

# EVALUATION
lr_a2 = accuracy_score(y_test, prediction) * 100
lr_p2 = precision_score(y_test, prediction) * 100
lr_r2 = recall_score(y_test, prediction) * 100
lr_f12 = f1_score(y_test, prediction) * 100

# Print Confusion Matrix
cm6 = confusion_matrix(y_test, prediction)
cmd6 = ConfusionMatrixDisplay(cm6, display_labels=lr2.classes_)
print(cm6)

print("Logistic Regression (TFIDF Vectorizer):")
print("Accuracy:", lr_a2)
print("Precision:", lr_p2)
print("Recall:", lr_r2)
print("F1-Score:", lr_f12)
print()

# COMPARING ACCURACY
model_accuracy = {'MNB': [round(mnb_a1), round(mnb_a2)],
                  'SVM': [round(svm_a1), round(svm_a2)],
                  'LR': [round(lr_a1), round(lr_a2)]
                  }
ma = pd.DataFrame(model_accuracy, columns=['MNB', 'SVM', 'LR'], index=['Count Vectorizer', 'TFIDF Vectorizer'])
ma

# COMPARING PRECISION
model_precision = {'MNB': [round(mnb_p1), round(mnb_p2)],
                   'SVM': [round(svm_p1), round(svm_p2)],
                   'LR': [round(lr_p1), round(lr_p2)]
                   }
mp = pd.DataFrame(model_precision, columns=['MNB', 'SVM', 'LR'], index=['Count Vectorizer', 'TFIDF Vectorizer'])
mp

# COMPARING RECALL
model_recall = {'MNB': [round(mnb_r1), round(mnb_r2)],
                'SVM': [round(svm_r1), round(svm_r2)],
                'LR': [round(lr_r1), round(lr_r2)]
                }
mr = pd.DataFrame(model_recall, columns=['MNB', 'SVM', 'LR'], index=['Count Vectorizer', 'TFIDF Vectorizer'])
mr

# COMPARING F1 SCORE
model_f1 = {'MNB': [round(mnb_f11), round(mnb_f12)],
             'SVM': [round(svm_f11), round(svm_f12)],
             'LR': [round(lr_f11), round(lr_f12)]
             }
mf1 = pd.DataFrame(model_f1, columns=['MNB', 'SVM', 'LR'], index=['Count Vectorizer', 'TFIDF Vectorizer'])
mf1

# SAVING THE BEST MODEL WITH ITS RESPECTIVE VECTORIZER
pickle.dump(lr1, open('best_model.pkl', 'wb'))
pickle.dump(count_vectorizer, open('count_vectorizer.pkl', 'wb'))
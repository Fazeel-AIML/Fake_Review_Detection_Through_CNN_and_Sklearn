# LIBRARIES
import pandas as pd
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.simplefilter("ignore")

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# LOADING DATASETS
df = pd.read_csv("D:\\Important\\Fake_Detection\\_fake_review_detector-main\\_fake_review_detector-main\\data and pickle files\\Dataset\\cleaned_data.csv", encoding="latin1")
toCheck = pd.read_csv("D:\\Important\\Fake_Detection\\_fake_review_detector-main\\_fake_review_detector-main\\data and pickle files\\Dataset\\updated_data.csv", encoding="latin1")

# REMOVE MAX
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# DROP EXTRA COLUMNS
df.drop(['Unnamed: 0'], axis=1, inplace=True)
toCheck.drop(['Unnamed: 0'], axis=1, inplace=True)

# CHECKING NULL VALUES
df = df.dropna(how='any', axis=0)

# ASSIGN THE VARIABLES
X = df['review_text']  # input var
y = df['verified_purchase']  # target var

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    df['review_text'], df['verified_purchase'], test_size=0.3, random_state=42)

# TOKENIZE TEXT
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# CONVERT TEXT TO SEQUENCES
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# ADD PADDING TO ENSURE CONSISTENT SEQUENCE LENGTH
vocab_size = len(tokenizer.word_index) + 1
max_len = 100
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# BUILDING THE CNN MODEL
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# COMPILE THE MODEL
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TRAIN THE MODEL
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# EVALUATE THE MODEL
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# PREDICT ON TEST DATA
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# PRINT EVALUATION METRICS
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred) * 100:.2f}%")

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# CLASSIFICATION REPORT
print("Classification Report:")
print(classification_report(y_test, y_pred))

# SAVE THE MODEL AND TOKENIZER
model.save('D:\\Important\\Fake_Detection\\_fake_review_detector-main\\_fake_review_detector-main\\data and pickle files\\CNN\\Model.h5')
with open('D:\\Important\\Fake_Detection\\_fake_review_detector-main\\_fake_review_detector-main\\data and pickle files\\CNN\\tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
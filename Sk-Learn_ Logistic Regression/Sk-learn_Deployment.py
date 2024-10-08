import re
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QDialog, QPlainTextEdit, QLabel, QPushButton, QGridLayout, QComboBox
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt 
from PyQt5.uic import loadUi
nltk.download('stopwords')

# LOAD THE MODEL AND VECTORIZER
with open('Sk-Learn_ Logistic Regression\Model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Sk-Learn_ Logistic Regression\Vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# TEXT PREPROCESSING
sw = set(stopwords.words('english'))

def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()

    cleaned = []
    stemmed = []

    for token in tokens:
        if token not in sw:
            cleaned.append(token)

    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)


# TEXT CLASSIFICATION
def text_classification():
    text = plain_text_edit.toPlainText()
    if len(text) < 1:
        result_label.setText(" ")
        result_label.setStyleSheet("QLabel { color: black; }")
    else:
        cleaned_review = text_preprocessing(text)
        sequence = vectorizer.transform([cleaned_review])
        prediction = model.predict(sequence)

        if prediction[0] == 1:
            result_text = "The review entered is Fraudulent."
            result_color = "red"
        else:
            result_text = "The review entered is Legitimate."
            result_color = "green"

        result_label.setText(result_text)
        result_label.setStyleSheet(f"QLabel {{ color: {result_color}; }}")

    # Load the images and create QPixmap objects
    result_evaluation_image = QPixmap('Sk-Learn_ Logistic Regression\confusion_matrix.png')
    confusion_matrix_image = QPixmap('Sk-Learn_ Logistic Regression\\result.png')

    # Adjust the size of the images
    result_evaluation_image = result_evaluation_image.scaled(1080, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.SmoothTransformation)
    confusion_matrix_image = confusion_matrix_image.scaled(1080, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.SmoothTransformation)

    # Create QLabel objects for the images
    result_evaluation_label = QLabel()
    confusion_matrix_label = QLabel()

    # Set the QPixmap objects as the images for the QLabel widgets
    result_evaluation_label.setPixmap(result_evaluation_image)
    confusion_matrix_label.setPixmap(confusion_matrix_image)

    # Set the alignment of the QLabel widgets
    result_evaluation_label.setAlignment(Qt.AlignCenter)
    confusion_matrix_label.setAlignment(Qt.AlignCenter)

    # Remove any existing widgets from the layout
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    # Get the selected option from the combo box
    selected_option = combo_box.currentText()

    if selected_option == "Result Evaluation Graph":
        # Add the QLabel widget for the result evaluation image to the layout
        layout.addWidget(confusion_matrix_label, 1, 1, Qt.AlignBottom)
    elif selected_option == "Confusion Matrix Graph":
        # Add the QLabel widget for the confusion matrix image to the layout
        layout.addWidget(result_evaluation_label, 1, 1, Qt.AlignBottom)

    # Set the layout for the dialog
    dialog.setLayout(layout)


if __name__ == "__main__":
    # Create the application instance
    app = QApplication([])
    # Load the UI file
    ui_file = 'Review_sk.ui'
    dialog = QDialog()
    loadUi(ui_file, dialog)

    # Get references to the UI elements
    plain_text_edit = dialog.findChild(QPlainTextEdit, "Review_AREA")
    result_label = dialog.findChild(QLabel, "Label_2")
    check_button = dialog.findChild(QPushButton, "Button")
    combo_box = dialog.findChild(QComboBox, "comboBox")

    # Add options to the combo box
    combo_box.addItem("Result Evaluation Graph")
    combo_box.addItem("Confusion Matrix Graph")

    # Create a QGridLayout to hold the images
    layout = QGridLayout()

    # Connect the text_classification function to the button's clicked event
    check_button.clicked.connect(text_classification)

    # Connect the text_classification function to the combo box's currentIndexChanged event
    combo_box.currentIndexChanged.connect(text_classification)

    dialog.setWindowTitle("Fake Review Detection")

    # Call the text_classification function to show the initial graph
    text_classification()

    # Show the dialog
    dialog.exec()
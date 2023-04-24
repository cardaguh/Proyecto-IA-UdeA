# Importamos las librerías necesarias
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargamos los datos en un DataFrame
data = pd.read_csv('Dataset_project.csv', encoding='latin-1', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])

# Eliminamos las columnas que no vamos a utilizar
data.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)

# Convertimos los valores de target (0, 2, 4) a (-1, 0, 1)
data['target'] = data['target'].apply(lambda x: -1 if x==0 else (0 if x==2 else 1))

# Preprocesamiento de los datos
stop_words = set(stopwords.words('english'))
def preprocess(text):
    # Eliminamos las menciones (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Eliminamos las URLs
    text = re.sub(r'http\S+', '', text)
    # Eliminamos los caracteres especiales y los números
    text = re.sub(r'\W+', ' ', text)
    # Convertimos a minúsculas
    text = text.lower()
    # Tokenizamos y eliminamos las palabras vacías
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    # Convertimos las palabras a su raíz
    stemmer = nltk.PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Unimos las palabras en una cadena de texto
    return ' '.join(words)

data['text'] = data['text'].apply(preprocess)

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# Creamos una matriz de términos de frecuencia de documento utilizando CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Entrenamos un clasificador Naive Bayes con los datos de entrenamiento
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Realizamos predicciones sobre los datos de prueba
y_pred = clf.predict(X_test_counts)

# Evaluamos el desempeño del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

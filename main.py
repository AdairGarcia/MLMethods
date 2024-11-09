import spacy as sp
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Cargar modelo de lenguaje español
nlp = sp.load("es_core_news_sm")

# funcion para extraer titulo y contenido del raw data corpus
# y tambien recupera el target
def extraer_datos(ruta):
    df = pd.read_csv(ruta)
    d = df[['Title', 'Content']]
    target = df['Section']
    return d, target

# funcion para concatenar dos columnas de datos en una sola para todas las columnas
# y crear el nuevo archivo cvs
def generar_raw_corpus(data, target):
    # Verificar si el archivo ya existe
    try:
        df = pd.read_csv('corpus/raw_data_corpus.csv')
        proporcion_corpus(df)
    except FileNotFoundError:
        new_df = pd.DataFrame()
        new_df['Title + Content'] = data['Title'] + ' ' + data['Content']
        new_df['Target'] = target
        new_df.to_csv('corpus/raw_data_corpus.csv', index=False)
        proporcion_corpus(new_df)

# funcion para saber la proporcion de los target en los datos
def stratification(data):
    return data['Target'].value_counts(normalize=True)

# funcion para separar el 80% de los datos para entrenamiento y el 20% para test
# esta funcion utiliza la proporcion de los target para mantener la proporcion en los datos
# en el entrenamiento y test
def proporcion_corpus(data):
    # Verificar si los archivos ya existen
    try:
        train = pd.read_csv('corpus/raw_train_corpus.csv')
        test = pd.read_csv('corpus/raw_test_corpus.csv')
        print('Los corpus ya existen')
        print(f'Proporcion de los datos crudos: {stratification(data)}')
        print(f'Proporcion de los datos de entrenamiento: {stratification(train)}')
        print(f'Proporcion de los datos de test: {stratification(test)}')
    except FileNotFoundError:
        proportion = stratification(data)
        train = pd.DataFrame()
        test = pd.DataFrame()

        for target in proportion.index:
            target_data = data[data['Target'] == target]
            target_data = target_data.sample(frac=1)
            target_train = target_data.iloc[:int(len(target_data) * 0.8)]
            target_test = target_data.iloc[int(len(target_data) * 0.8):]
            train = pd.concat([train, target_train])
            test = pd.concat([test, target_test])

        train.to_csv('corpus/raw_train_corpus.csv', index=False)
        test.to_csv('corpus/raw_test_corpus.csv', index=False)

        print(f'Proporcion de los datos crudos: {stratification(data)}')
        print(f'Proporcion de los datos de entrenamiento: {stratification(train)}')
        print(f'Proporcion de los datos de test: {stratification(test)}')

        print('Raw Corpus generados con exito')

# Funcion para normalizar los datos
def normalizer(string):
    if not isinstance(string, str):
        return ""

    doc = nlp(string)
    postaggin = ['DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ']

    no_stop_words_string = ""

    for token in doc:
        if not(token.pos_ in postaggin) and (token.pos_ != "PUNCT"):
            no_stop_words_string += token.lemma_ + " "
        elif token.pos_ == "PUNCT" and token.i == len(doc) - 1:
            no_stop_words_string = no_stop_words_string[:-1]
            no_stop_words_string += token.lemma_

    return no_stop_words_string

# funcion para normalizar los datos y crear el archivo cvs
def generar_normalized_corpus():
    df = pd.read_csv('corpus/raw_train_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(normalizer)
    df.to_csv('corpus/normalized_train_corpus.csv', index=False)

    df = pd.read_csv('corpus/raw_test_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(normalizer)
    df.to_csv('corpus/normalized_test_corpus.csv', index=False)

def frequency_vectorize(corpus):
    vector = CountVectorizer()
    x = vector.fit_transform(corpus)
    return vector, x

def binary_vectorize(corpus):
    vector = CountVectorizer(binary=True)
    x = vector.fit_transform(corpus)
    return vector, x

def tfidf_vectorize(corpus):
    vector = TfidfVectorizer()
    x = vector.fit_transform(corpus)
    return vector, x

# TODO: Implementar la funcion
def embeddings_vectorize(corpus):
    pass
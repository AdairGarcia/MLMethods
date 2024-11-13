import pickle

import pandas as pd
import spacy as sp
from sklearn.decomposition import TruncatedSVD
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
        data['Title'] = data['Title'].fillna('')
        data['Content'] = data['Content'].fillna('')
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

def remove_stop_words(string):
    if not isinstance(string, str):
        return ""

    doc = nlp(string)
    postaggin = ['DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 'PUNCT']

    no_stop_words_string = ""

    for token in doc:
        if not(token.pos_ in postaggin):
            no_stop_words_string += token.text+ " "

    return no_stop_words_string

def lemmatiser(string):
    if not isinstance(string, str):
        return ""

    doc = nlp(string)

    lemmatised_string = ""

    for token in doc:
        lemmatised_string += token.lemma_ + " "

    return lemmatised_string

# funcion para normalizar los datos y crear el archivo cvs
def generar_normalized_corpus():
    df = pd.read_csv('corpus/raw_train_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(normalizer)
    df.to_csv('corpus/normalized_train_corpus.csv', index=False)

    df = pd.read_csv('corpus/raw_test_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(normalizer)
    df.to_csv('corpus/normalized_test_corpus.csv', index=False)

def generar_lemmatized_corpus():
    df = pd.read_csv('corpus/raw_train_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(lemmatiser)
    df.to_csv('corpus/lemmatized_train_corpus.csv', index=False)

    df = pd.read_csv('corpus/raw_test_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(lemmatiser)
    df.to_csv('corpus/lemmatized_test_corpus.csv', index=False)

def generar_stop_words_removed_corpus():
    df = pd.read_csv('corpus/raw_train_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(remove_stop_words)
    df.to_csv('corpus/no_stop_words_train_corpus.csv', index=False)

    df = pd.read_csv('corpus/raw_test_corpus.csv')
    df['Title + Content'] = df['Title + Content'].apply(remove_stop_words)
    df.to_csv('corpus/no_stop_words_test_corpus.csv', index=False)

# funcion para la representacion de los datos en frecuencia
def frequency_vectorize(corpus):
    vector = CountVectorizer()
    x = vector.fit_transform(corpus)
    return x

# funcion para la representacion de los datos en binario
def binary_vectorize(corpus):
    vector = CountVectorizer(binary=True)
    x = vector.fit_transform(corpus)
    return x

# funcion para la representacion de los datos en tfidf
def tfidf_vectorize(corpus):
    vector = TfidfVectorizer()
    x = vector.fit_transform(corpus)
    return x

# Función para aplicar SVD a una representación
def apply_svd(x, n_components=50):
    svd = TruncatedSVD(n_components=n_components)
    x_reduced = svd.fit_transform(x)
    return x_reduced

# Función para generar y guardar las representaciones vectorizadas
def generar_vectorized_data():
    train = pd.read_csv('corpus/normalized_train_corpus.csv')
    test = pd.read_csv('corpus/normalized_test_corpus.csv')

    # Fill NaN values with empty strings
    train['Title + Content'] = train['Title + Content'].fillna('')
    test['Title + Content'] = test['Title + Content'].fillna('')

    train_corpus = train['Title + Content'].tolist()
    test_corpus = test['Title + Content'].tolist()

    vectorization_methods = {
        'freq': frequency_vectorize,
        'binary': binary_vectorize,
        'tfidf': tfidf_vectorize,
    }

    for method_name, vectorize in vectorization_methods.items():
        x_train = vectorize(train_corpus)
        x_test = vectorize(test_corpus)


        # Guardar las representaciones vectorizadas
        with open(f'unigrams/{method_name}_train.pkl', 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(x_train, f)
        with open(f'unigrams/{method_name}_test.pkl', 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(x_test, f)

        # Aplicar SVD y guardar las representaciones reducidas
        x_train_svd = apply_svd(x_train)
        x_test_svd = apply_svd(x_test)

        with open(f'unigrams/{method_name}_train_svd.pkl', 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(x_train_svd, f)
        with open(f'unigrams/{method_name}_test_svd.pkl', 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(x_test_svd, f)

    print(f'Representaciones vectorizadas guardadas con éxito')

if __name__ == '__main__':
    data, target = extraer_datos('corpus/raw data corpus.csv')
    generar_raw_corpus(data, target)
    generar_normalized_corpus()
    generar_lemmatized_corpus()
    generar_stop_words_removed_corpus()

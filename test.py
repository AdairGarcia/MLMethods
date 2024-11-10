from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding

from methods import *


def get_x_y(mode):
    if mode == 'raw':
        return get_data('corpus/raw_train_corpus.csv'), get_data('corpus/raw_test_corpus.csv')
    elif mode == 'normalized':
        return get_data('corpus/normalized_train_corpus.csv'), get_data('corpus/normalized_test_corpus.csv')
    elif mode == 'lemmatized':
        return get_data('corpus/lemmatized_train_corpus.csv'), get_data('corpus/lemmatized_test_corpus.csv')
    elif mode == 'no_stop_words':
        return get_data('corpus/no_stop_words_train_corpus.csv'), get_data('corpus/no_stop_words_test_corpus.csv')

def fill_na(x_train, x_test):
    x_train = x_train.fillna('')
    x_test = x_test.fillna('')
    return x_train, x_test

def test():
    modes = ['raw', 'normalized', 'lemmatized', 'no_stop_words']
    models = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': MultinomialNB(),
        'Support Vector Machine': SVC(),
        'Multilayer Perceptron': MLPClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
    }
    text_representations = {
        'Frequency': CountVectorizer(),
        'Binary': CountVectorizer(binary=True),
        'TF-IDF': TfidfVectorizer(),
    }

    for mode in modes:
        print(f'#######################################-{mode.upper()}-#######################################')
        (x_train, y_train), (x_test, y_test) = get_x_y(mode)
        x_train, x_test = fill_na(x_train, x_test)

        for text_representation_name, text_representation in text_representations.items():
            for model_name, model in models.items():
                if model_name == 'Naive Bayes' and text_representation_name == 'TF-IDF':
                    print(f'-------------------------{text_representation_name} - {model_name}-------------------------{mode.upper()}')
                    pipelines(x_train, y_train, x_test, y_test, text_representation, model, 1)
                    print()

                print(f'-------------------------{text_representation_name} - {model_name}-------------------------{mode.upper()}')
                pipelines(x_train, y_train, x_test, y_test, text_representation, model)
                print()
            print()

if __name__ == '__main__':
    test()
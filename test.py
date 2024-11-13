import time

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import compute_class_weight
import numpy as np

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

def test():
    start_time = time.time()
    modes = ['raw', 'lemmatized', 'no_stop_words', 'normalized']
    models = {
        # 'Logistic Regression': LogisticRegression(),
        # 'Naive Bayes': MultinomialNB(),
        # 'Support Vector Machine': SVC(),
        'Multilayer Perceptron': MLPClassifier(early_stopping=True),
        # 'Random Forest': RandomForestClassifier(),
        # 'Gradient Boosting': GradientBoostingClassifier(),
    }
    text_representations = {
        'Frequency': CountVectorizer(),
        'Binary': CountVectorizer(binary=True),
        'TF-IDF': TfidfVectorizer(),
    }
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1, 10, 100]
        },
        'Naive Bayes': {
            'classifier__alpha': [0.1, 0.5, 1.0]
        },
        'Support Vector Machine': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf']
        },
        'Multilayer Perceptron': {
            'classifier__hidden_layer_sizes': [(50,), (100,)],
            'classifier__activation': ['tanh', 'relu'],
            'classifier__alpha': [0.01]
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_features': ['sqrt', 'log2', None]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    }

    for mode in modes:
        print(f'#######################################-{mode.upper()}-#######################################')
        (x_train, y_train), (x_test, y_test) = get_x_y(mode)

        for text_representation_name, text_representation in text_representations.items():
            for model_name, model in models.items():
                # if model_name == 'Naive Bayes' and text_representation_name == 'TF-IDF':
                #     print(f'-------------------------{text_representation_name} - {model_name}-------------------------{mode.upper()}')
                #     pipelines(x_train, y_train, x_test, y_test, text_representation, model, 1)
                #     print()
                #
                # print(f'-------------------------{text_representation_name} - {model_name}-------------------------{mode.upper()}')
                # pipelines(x_train, y_train, x_test, y_test, text_representation, model)
                # print()
                pipeline = Pipeline(
                    [('vectorizer', text_representation), ('classifier', model)]
                )

                param_grid = param_grids[model_name]
                param_grid.update({
                    'vectorizer__max_features': [5000],
                    'vectorizer__ngram_range': [(1, 1)],
                    'vectorizer__max_df': [0.75, 0.85, 0.95],
                    'vectorizer__min_df': [1, 2, 5]
                })

                # Compute class weights
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

                # Update the classifier with class weights
                pipeline.set_params(classifier__class_weight=class_weight_dict)

                grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
                grid_search.fit(x_train, y_train)

                print(f'-------------------------{text_representation_name} - {model_name}-------------------------{mode.upper()}')
                print(f'Mejores parametros {text_representation_name} - {model_name}: {grid_search.best_params_}')
                print(f'Mejor macro avg f1-score: {grid_search.best_score_}')

                y_pred = grid_search.predict(x_test)
                print(classification_report(y_test, y_pred, zero_division=0))
                print()
            print()
    end_time = time.time()
    print(f'Tiempo de ejecución: {end_time - start_time} segundos')


if __name__ == '__main__':
    test()
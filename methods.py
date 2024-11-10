import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


# funcion que utiliza pipelines de spacy para
# el uso de modelos de machine learning
def pipelines(x_train, y_train, x_test, y_test, text_representation, classifier):
    pipe = Pipeline([('text_representation', text_representation), ('classifier', classifier)])
    print(pipe)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    print(classification_report(y_test, y_pred))

# funcion para recuperar el target de los datos de test o train
def get_data(ruta):
    corpus = pd.read_csv(ruta)
    return corpus['Title + Content'], corpus['Target']



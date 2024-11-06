import pandas as pd

# funcion para extraer titulo y contenido del raw data corpus
# y tambien recupera el target
def extraer_datos(ruta):
    df = pd.read_csv(ruta)
    d = df[['Title', 'Content']]
    target = df['Section']
    return d, target

# funcion para concatenar dos columnas de datos en una sola para todas las columnas
# y crear el nuevo archivo cvs
def concatenar_columnas(data, target):
    data['Title + Content'] = data['Title'] + ' ' + data['Content']
    # se crea un nuevo dataframe con las columnas concatenadas y agregando su correspondiente target
    new_data_df = pd.DataFrame(data['Title + Content'], columns=['Title + Content', 'Target'])

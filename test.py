from main import extraer_datos, generar_raw_corpus, generar_normalized_corpus

data = extraer_datos('./corpus/raw data corpus.csv')
generar_raw_corpus(data[0], data[1])
generar_normalized_corpus()
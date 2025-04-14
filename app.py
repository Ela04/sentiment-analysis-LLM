from transformers import pipeline
""""
# Carga modelo de análisis de sentimiento multilingüe
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Solicita texto al usuario
texto = input("Escribe un texto para analizar su sentimiento: ")

# Ejecuta el análisis
resultado = classifier(texto)

# Muestra resultado
print("\nResultado del análisis:")
print(resultado)

"""
###############################################################################
# Elige este modelo (mejor para sentimiento general)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)

texto = input("Escribe un texto para analizar su sentimiento: ")
resultado = classifier(texto)
sentimiento = resultado[0]['label']
confianza = round(resultado[0]['score'] * 100, 2)

print(f"\n💬 Resultado: {sentimiento} ({confianza}% de confianza)")
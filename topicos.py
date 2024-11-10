import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Crear un DataFrame de ejemplo
data = {
    'state': ['Texas', 'California', 'Texas', 'California', 'Texas', 'California'],
    'text': [
        'The weather in Texas is hot and dry.',
        'California is known for its beaches and parks.',
        'Texas is a large state with many people.',
        'California has a lot of tech companies.',
        'The economy in Texas is booming.',
        'California has a huge entertainment industry.'
    ]
}

df = pd.DataFrame(data)

# 1. Preprocesamiento de Texto: Aquí puedes hacer más trabajo de limpieza si es necesario
# Por ahora, vamos a usar el texto tal cual, pero podrías eliminar stop words, lematizar, etc.

# 2. Crear una función para hacer el análisis de tópicos con LDA
def lda_analysis_by_state(df, n_topics=3):
    # Vectorizar el texto usando TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    
    # Agrupar por 'state' y aplicar el análisis de tópicos
    states = df['state'].unique()
    
    lda_results = {}
    
    for state in states:
        # Filtrar los textos para el estado actual
        state_texts = df[df['state'] == state]['text']
        
        # Vectorizar los textos
        X = vectorizer.fit_transform(state_texts)
        
        # Aplicar LDA para encontrar n_topics tópicos
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        
        # Almacenar los resultados de LDA
        lda_results[state] = {
            'model': lda,
            'vectorizer': vectorizer
        }
    
    return lda_results

# 3. Aplicar el análisis de tópicos para cada 'state'
lda_results = lda_analysis_by_state(df, n_topics=3)

# 4. Mostrar los términos más importantes para cada tópico de cada estado
def print_top_words(lda_results, n_words=10):
    for state, result in lda_results.items():
        print(f"State: {state}")
        lda = result['model']
        vectorizer = result['vectorizer']
        
        print("Top words per topic:")
        for topic_idx, topic in enumerate(lda.components_):
            print(f"  Topic #{topic_idx + 1}:")
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_words - 1:-1]]
            print(f"    {' '.join(top_words)}")
        print("\n")

# Imprimir los resultados
print_top_words(lda_results)

# 5. Crear Nubes de Palabras para cada 'state'
def plot_wordclouds(lda_results):
    for state, result in lda_results.items():
        lda = result['model']
        vectorizer = result['vectorizer']
        
        # Crear la nube de palabras para cada tópico
        for topic_idx, topic in enumerate(lda.components_):
            topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-20 - 1:-1]]
            wordcloud = WordCloud(width=800, height=400).generate(' '.join(topic_words))
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f"WordCloud for {state} - Topic #{topic_idx + 1}")
            plt.axis('off')
            plt.show()

# Mostrar las nubes de palabras
plot_wordclouds(lda_results)

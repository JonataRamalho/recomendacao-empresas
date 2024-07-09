import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Função para normalizar as notas das categorias
def normalizar_dados(df, categorias):
    scaler = StandardScaler()
    df[categorias] = scaler.fit_transform(df[categorias])
    return df, scaler

# Função para normalizar os pesos
def normalizar_pesos(pesos):
    soma_pesos = sum(pesos)
    return [peso / soma_pesos for peso in pesos]

# Dados de entrada
categorias_interesse = ['Oportunidades de carreira', 'Remuneração e benefícios', 'Cultura e valores']
pesos = [5, 3, 2]  # Pesos atribuídos pelo usuário para cada categoria de interesse

# Normalizar os pesos
pesos = normalizar_pesos(pesos)

empresas_data = {
    'Empresa': ['Tractian', 'XP Inc', 'Empresa C', 'Empresa D', 'Empresa E'],
    'Nota Geral': [4.1, 3.5, 4.0, 3.8, 3.9],
    'Oportunidades de carreira': [4.4, 3.7, 4.2, 3.9, 4.1],
    'Remuneração e benefícios': [4.3, 3.8, 4.1, 3.6, 4.0],
    'Cultura e valores': [3.9, 3.3, 4.0, 3.7, 3.8],
    'Diversidade e inclusão': [3.9, 3.3, 4.1, 3.5, 3.7],
    'Alta liderança': [3.7, 3.1, 3.9, 3.6, 3.8],
    'Qualidade de vida': [3.5, 2.8, 3.7, 3.2, 3.4]
}

# Convertendo o dicionário para um DataFrame
empresas = pd.DataFrame(empresas_data)

# Normalizar as notas das categorias
empresas, scaler = normalizar_dados(empresas, categorias_interesse)

# Aplicar os pesos às categorias de interesse
for i, categoria in enumerate(categorias_interesse):
    empresas[categoria] *= pesos[i]

# Selecionar as colunas de interesse (categorias) para o KNN
X = empresas[categorias_interesse].values

# Treinar o modelo KNN
knn = NearestNeighbors(n_neighbors=len(empresas), algorithm='auto')
knn.fit(X)

# Simulação de novo usuário com notas desejadas para as categorias de interesse
novo_usuario = np.array([[4.4, 4.3, 3.9]])  # Notas desejadas pelo usuário para as categorias de interesse
novo_usuario = scaler.transform(novo_usuario)  # Normalizar as notas do novo usuário

# Aplicar os pesos às notas do novo usuário
novo_usuario = novo_usuario * pesos

# Encontrar as empresas mais semelhantes
distances, indices = knn.kneighbors(novo_usuario)

# Obter as empresas recomendadas
empresas_recomendadas = empresas.iloc[indices[0]]

# Exibir a melhor empresa
melhor_empresa = empresas_recomendadas.iloc[0]
print(f"A melhor empresa é: {melhor_empresa['Empresa']}")
print(f"Nota Geral: {melhor_empresa['Nota Geral']}")
print("\nCategorias e notas:")
for categoria in categorias_interesse:
    print(f"{categoria}: {melhor_empresa[categoria]}")x

# Exibir o ranking das empresas
print("\nRanking das empresas:")
for i, row in empresas_recomendadas.iterrows():
    print(f"{row['Empresa']} - Similaridade: {1 - distances[0][i]:.2f}, Nota Geral: {row['Nota Geral']}")

# Usuário só deve passar a categoria, não pode peso e nem nota
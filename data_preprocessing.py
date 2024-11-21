import kagglehub

path = kagglehub.dataset_download("jeremylarcher/canadian-house-prices-for-top-cities")

import os
print(os.listdir(path))

file_name = "HouseListings-Top45Cities-10292023-kaggle.csv"  # Substitua pelo nome do arquivo listado
file_path = os.path.join(path, file_name)

import pandas as pd

data = pd.read_csv(file_path, encoding='latin-1')  # Tente 'latin-1' ou 'cp1252'

print(data.head())  # Primeiras linhas

print(data.info())  # Informações sobre o dataset

# Valores únicos da coluna 'City'
unique_cities = data['City'].unique()
print(unique_cities)

print(data.describe())


sns.histplot(data['Price'])

pip install tensorflow_decision_forests


# Filtrando os dados onde a coluna 'City' é igual a 'Toronto'
toronto_data = data.query('City == "Toronto"')

# Exibindo os primeiros registros para verificar
print(toronto_data.head())

# Selecionando as variáveis relevantes para o modelo
features = ['Number_Beds', 'Number_Baths', 'Median_Family_Income']
X = toronto_data[features]  # Variáveis independentes
y = toronto_data['Price']   # Variável dependente (preço)





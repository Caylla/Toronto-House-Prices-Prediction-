
pip install tensorflow_decision_forests

# Filtrando os dados onde a coluna 'City' é igual a 'Toronto'
toronto_data = data.query('City == "Toronto"')

# Exibindo os primeiros registros para verificar
print(toronto_data.head())

# Selecionando as variáveis relevantes para o modelo
features = ['Number_Beds', 'Number_Baths', 'Median_Family_Income']
X = toronto_data[features]  # Variáveis independentes
y = toronto_data['Price']   # Variável dependente (preço)

# Concatenação dos dados em um único DataFrame
df = pd.concat([X, y], axis=1)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando DataFrames para treino e teste
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Convertendo os DataFrames para o formato TensorFlow Dataset
train_data = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, task=tfdf.keras.Task.REGRESSION, label="Price")
test_data = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, task=tfdf.keras.Task.REGRESSION, label="Price")


# Criando o modelo de Random Forest
model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)


# Treinando o modelo
model.fit(train_data)



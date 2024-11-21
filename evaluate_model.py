import tensorflow_decision_forests as tfdf
from sklearn.metrics import mean_squared_error, r2_score

# Fazendo previsões nos dados de teste
predictions = model.predict(test_data)

# Convertendo as previsões para uma lista
predictions = [pred[0] for pred in predictions]

# Import the necessary libraries
from sklearn.metrics import mean_squared_error, r2_score 

# Avaliando o modelo
mse = mean_squared_error(y_test, predictions)  # Erro quadrático médio
rmse = mse**0.5  # Raiz do erro quadrático médio
r2 = r2_score(y_test, predictions)  # R² (coeficiente de determinação)

# Exibindo as métricas
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R²: {r2}")

# Comparando as previsões com os valores reais
for true, pred in zip(y_test, predictions):
    print(f"True: {true}, Predicted: {pred}")



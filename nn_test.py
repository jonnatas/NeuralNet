from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy
import os

numpy.random.seed(7)

#carrregando dataser
dataset = numpy.loadtxt("diabetes.csv", delimiter=",")

# Separando os dados em entrada e saida

X = dataset[:, 0:8]
Y = dataset[:, 8]

#Aplicando scaler pra normalizar dados

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

#Separando dados pra treino e teste

X_train, X_test, Y_train, Y_test = train_test_split(X_scaler, Y, test_size=0.02, random_state=10)


#Definindo o modelo

# 8 entradas
# Camadas Fully connected
# 12 , 5 , 1 neuronios por camada
# Pesos definidos aleatóriamente [0, 0.05]
# Funções de ativação rectifier, rectifier, sigmoid
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compilando o modelo 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Treinando o modelo 
model.fit(X_train, Y_train, epochs=150, batch_size=10)

#Avaliando o modelo
scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Predizendo

predictions = model.predict(X_test)

rounded = [round(x[0]) for x in predictions]
print(rounded)
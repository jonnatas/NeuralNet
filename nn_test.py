from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

#carrregando dataser
dataset = numpy.loadtxt("diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]
# Separando os dados em entrada e saida


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
model.fit(X, Y, epochs=150, batch_size=10)

#Avaliando o modelo
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Predict

predictions = model.predict(X)

rounded = [round(x[0]) for x in predictions]
print(rounded)
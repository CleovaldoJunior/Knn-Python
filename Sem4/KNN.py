import pandas as pd
from sklearn.model_selection import train_test_split

# Essa função faz justamente o cálculo da distância entre 2 linhas, ponto a ponto
# Retornando a raiz quadrada da soma das distâncias entre cada ponto
def euclidian_distance(linha1, linha2):
    distancia = 0
    for i in range(len(linha1)):
        distancia += (linha1[i]-linha2[i])**2
    return distancia**(1/2)

# Essa função vai retornar as N menores distâncias entre cada linha do teste
# com cada linha do treino, de forma que as distâncias são calculadas usando
# a função de distância euclidiana
def treino(teste, treino_x, treino_y, n):
    vizinhos_total = []
    for teste_atual in teste:
        distancias = []
        i = 0
        for treino_atual in treino_x: #Para cada linha do teste percorre todas as linha do treino
            d_atual = euclidian_distance(teste_atual, treino_atual) #Calcula a distância euclidiana
            distancias.append([list(treino_atual), d_atual, treino_y[i]])
            i+=1
        distancias.sort(key=lambda tup:tup[1]) #Ordena as distâncias de forma crescente
        vizinhos = distancias[:n] #Pega as n menores distâncias
        vizinhos_total.append(vizinhos[0])
    return vizinhos_total

# Essa função basicamente faz o predict das menores distâncias
# se o Y da menor distância for igual ao Y da linha então tem-se mais um acerto
def predict(teste_x, teste_y, treino_x, treino_y, n):
    v = treino(teste_x, treino_x, treino_y, n)
    y_pred = []
    acc = 0
    for i in range(len(v)):
        y_pred.append(v[i][2])
        if v[i][2] == teste_y[i]:
            acc+=1
    print(acc/len(teste_x)*100)
    return y_pred

dataset_url = 'wine.data'
data = pd.read_csv(dataset_url,names=['y','1','2','3','4','5','6','7','8','9','10','11','12','13'])
x = data.drop(['y','13'],axis=1)
y = data.y

# Usei o train_test_split apenas para separar de forma mais fácil o treino do teste no dataset
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,test_size=0.3, random_state=42)
y_pred = predict(teste_x=teste_x.values, teste_y=teste_y.values, treino_x=treino_x.values, treino_y=treino_y.values, n=1)

vet_acc = [0,0,0]
vet_qtd = [0,0,0]

for i in range(len(y_pred)):
    if y_pred[i] == list(teste_y)[i]:
        vet_acc[y_pred[i]-1] += 1
    vet_qtd[y_pred[i]-1] += 1
print(vet_acc)
print(vet_qtd)

som = 0
for i in range(len(vet_acc)):
    som+= vet_acc[i]*vet_qtd[i]
print(som/(sum(vet_qtd)))

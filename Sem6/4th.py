from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from statistics import NormalDist

#Função que usei para calcular o intervalo de confiança com 95%
def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

#Vai gerar os valores omissos para o dataset
def valores_omissos(dataset):
    omissos_por_coluna = []
    #Percorre as colunas do dataset
    for key in dataset.keys():
        #Vai pegar a lista dos valores mais frequentes
        lista_valores_maximos = list(dict(dataset[key].value_counts()))
        #Pego a moda para trocar os valores omissos pela mesma
        moda = lista_valores_maximos[0]
        #Se a moda for o próprio valor omisso
        #Eu seto o valor do atributo como 0
        if moda == "?":
            moda = 0
        #Guardo a moda em um vetor para poder reusar
        #a mesma no conjunto de teste
        omissos_por_coluna.append(int(moda))
        #Aqui eu dou o replace em cada coluna no valor omisso
        dataset[key].replace({"?": int(moda)}, inplace=True)
    return dataset, omissos_por_coluna

#Usa da lista de moda dos omissos do treino para corrigir
#os omissos do teste
def corrige_omissos_teste(teste, lista_omissos):
    i = 0
    #Percore as colunas
    for key in teste.keys():
        #Da replace em cada omisso de cada coluna pela sua respectiva
        #moda gerada pelo conjunto de treino
        teste[key].replace({"?": lista_omissos[i]}, inplace=True)
        i+=1
    return teste

atts = ["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","y"]

hungarian = pd.read_csv('processed.hungarian.csv', header=None, names=atts)
X = hungarian.drop("y", axis=1)
y = hungarian.y

scores = []
#Knn com k = 1 e euclidiano
knn = KNeighborsClassifier(n_neighbors=1,p=2)
for i in range(100):
    #Pego o treino e teste com 90% do dataset como sendo treino e estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, stratify=y)
    #Corrigo os valores omissos para o treio e recupero a lista de modas usadas
    X_train, omissos_treino = valores_omissos(X_train.copy())
    #X_test, omissos_treino = valores_omissos(X_test.copy())
    #Usa da lista de moda do conjunto de treino para retirar
    #Os valores omissos do conjunto de teste
    X_test = corrige_omissos_teste(X_test.copy(), omissos_treino)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores.append(round(score,3))
print("   Média do score:", sum(scores)/len(scores))
print(confidence_interval(scores))


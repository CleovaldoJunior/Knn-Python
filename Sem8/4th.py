import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def numerico(dataset):
    #Percorre as colunas
    for i in range(len(dataset.columns)):
        coluna = dataset.iloc[:,i].values
        set_coluna = list(set(coluna))
        d = {}
        c = 0
        #Para cada elemento da coluna atual
        for ele in set_coluna:
            #Se o elemento for string
            if isinstance(ele, str):
                #O elemento é trocado pelo seu index
                d[ele] = c
            c += 1
        #Basicamente trocando a coluna antiga pela nova
        for j in range(len(coluna)):
            if isinstance(coluna[j], str):
                coluna[j] = d[coluna[j]]
    return dataset

coluna = ['1', '2', '3', '4','5','6','classe']
car = pd.read_csv('car.data', names=coluna)
scale = numerico(car)

X = car.drop(['classe'], axis=1)
y = car.classe.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, shuffle=True)

clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X_train, y_train)

coluna = ['1', '2', '3', '4','5','6']
score = clf.score(X_test, y_test)

tree.plot_tree(clf, feature_names=coluna)
plt.show()

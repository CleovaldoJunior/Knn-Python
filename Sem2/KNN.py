import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

coluna = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('iris.data', header=None, names=coluna)
classe_iris = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['species_num'] = [classe_iris[i] for i in iris.species]
X = iris.drop(['species', 'species_num'], axis=1)
y = iris.species_num

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7,p=2)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(score*100)



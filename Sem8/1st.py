import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

coluna = ['classe', 'esquerda_peso', 'esquerda_dist', 'direita_peso', 'direita_dist']
scale = pd.read_csv('balance-scale.data', names=coluna)
X = scale.drop(['classe'], axis=1)
y = scale.classe

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, shuffle=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pred_array = clf.predict(X_test)
print(clf.score(X_test, y_test))
print(clf.feature_importances_)
score = clf.score(X_test, y_test)

tree.plot_tree(clf)
plt.show()

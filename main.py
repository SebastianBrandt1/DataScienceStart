import pandas as pd
from sklearn import tree

X = [[]]
y = []
X_test = [[]]
df_new = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
print(df_new)
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
prediction = clf.predict(X_test)

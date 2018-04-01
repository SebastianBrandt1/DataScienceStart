from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score


class Classifier(object):
    def __init__(self,method):
        self.clf_method = method
        if(self.clf_method == 'DecisionTree'):
            self.clf = tree.DecisionTreeClassifier()
        elif(self.clf_method == 'NearestNeighbor'):
            self.clf = neighbors.KNeighborsClassifier(3)
        elif (self.clf_method == 'GaussianProcess'):
            self.clf = gaussian_process.GaussianProcessClassifier()
        elif (self.clf_method == 'LinearSVM'):
            self.clf = svm.SVC(kernel='linear')
        elif (self.clf_method == 'RBFSVM'):
            self.clf = svm.SVC()
        elif (self.clf_method == 'NaiveBayes'):
            self.clf = naive_bayes.GaussianNB()

    def learn(self,X,y):
        self.clf.fit(X,y)

    def predict(self,X):
        return self.clf.predict(X)

    def methodGet(self):
        return self.clf_method



X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

X_test = [[183,72,47],[173,72,39],[154,47,37],[178,65,41]]
y_test = ['male','female','female','male']

aClassifier = []
classifier = Classifier('DecisionTree')
aClassifier.append(classifier)
classifier = Classifier('NearestNeighbor')
aClassifier.append(classifier)
classifier = Classifier('GaussianProcess')
aClassifier.append(classifier)
classifier = Classifier('LinearSVM')
aClassifier.append(classifier)
classifier = Classifier('RBFSVM')
aClassifier.append(classifier)
classifier = Classifier('NaiveBayes')
aClassifier.append(classifier)
prediction = {}
score = {}
print(y_test)
for cl in aClassifier:
    cl.learn(X,y)
    prediction[cl.methodGet()] = cl.predict(X_test)
    score[cl.methodGet()] = accuracy_score(y_test,prediction[cl.methodGet()])

print(prediction)
print(score)

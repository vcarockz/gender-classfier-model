#gender classifier model- determines male or female based on few features
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Dataset for gender classification
#[height,weight,shoe size in uk]
X = [[181,80,11],[172,73,12],[166,60,8],[191,94,12],[159,55,7],[177,70,10]]

Y = ['male','male','female','male','female','male']

# Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
prediction = clf.predict([[156,62,10]])
print(prediction)

#Classifer used is Gaussian Naive Bayes
gnb = GaussianNB()
gnb = gnb.fit(X,Y)
predict = gnb.predict([[156,62,10]])
print(predict)
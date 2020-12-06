from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.simplefilter("ignore")

#Visualize/plot data
data = sns.load_dataset("iris")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

plt.xlabel('Features')
plt.ylabel('Species')

pltX = data.loc[:, 'sepal_length']
pltY = data.loc[:,'species']
plt.scatter(pltX, pltY, color='blue', label='sepal_length')

pltX = data.loc[:, 'sepal_width']
pltY = data.loc[:,'species']
plt.scatter(pltX, pltY, color='green', label='sepal_width')

pltX = data.loc[:, 'petal_length']
pltY = data.loc[:,'species']
plt.scatter(pltX, pltY, color='red', label='petal_length')

pltX = data.loc[:, 'petal_width']
pltY = data.loc[:,'species']
plt.scatter(pltX, pltY, color='black', label='petal_width')

plt.legend(loc=4, prop={'size':8})
plt.show()


iris_data = load_iris()

x = iris_data.data
y = iris_data.target

kf = KFold(n_splits=5)

print("\n\n\n----------Logistic Regression----------\n\n")
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(predictions)
    print( classification_report(y_test, predictions) )
    print( "Accuracy : ", accuracy_score(y_test, predictions))


print("\n\n\n----------Adaptive Boosting Classification----------\n\n")
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(predictions)
    print( classification_report(y_test, predictions) )
    print( "Accuracy : ", accuracy_score(y_test, predictions))


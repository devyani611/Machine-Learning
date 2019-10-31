import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

#load the data
wine = pd.read_csv('winequality-white.csv', sep=';')
wine.head()
#Getting the column values
feature_cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']

#barplot of data
plt.figure(figsize=(10, 6))
plt.title("Count of different qualities")
sns.countplot(wine["quality"], palette="Set3")
wine["quality"].value_counts()

#data transformation
quality = wine["quality"].values
category = []
for num in quality:
    if num < 5:
        category.append("bad")
    elif num > 6:
        category.append("High")
    else:
        category.append("normal")

#Count for values of class        
print("Count of different features:\n", [(i, category.count(i)) for i in set(category)])        

#plot bar with transformed data
plt.figure(figsize=(10, 6))
plt.title("Count of different categories")
sns.countplot(category)
 
#Replacing the quality with category and categorising the data into features and label    
category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([wine, category], axis=1)
data.drop(columns="quality", axis=1, inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


#Splitting the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)

#calling classifier using entropy and fitting the training data
print("\nPredicting the quality using entropy index:-")
clf_entropy=DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=100)
clf_entropy.fit(X_train, y_train)

# check its performance on test
pred_dt1 = clf_entropy.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != pred_dt1).sum()
print('\nMisclassified samples: {}'.format(count_misclassified))
#classification report
print("\nClassification report:\n",classification_report(y_test, pred_dt1))
#calculating accuracy
print("The dtree model accuracy using entropy on Test data is %s" % accuracy_score(y_test, pred_dt1))

# relabel back : 0 means good, 1 for bad, 2 for normal for better visualization of confusion matrix
y_test_re = list(y_test)
for i in range(len(y_test_re)):
    if y_test_re[i] == 0:
        y_test_re[i] = "good"
    if y_test_re[i] == 1:
        y_test_re[i] = "bad"
    if y_test_re[i] == 2:
        y_test_re[i] = "normal"
pred_dt1_re = list(pred_dt1)
for i in range(len(pred_dt1_re)):
    if pred_dt1_re[i] == 0:
        pred_dt1_re[i] = "good"
    if pred_dt1_re[i] == 1:
        pred_dt1_re[i] = "bad"
    if pred_dt1_re[i] == 2:
        pred_dt1_re[i] = "normal"
y_actu = pd.Series(y_test_re, name='Actual')
y_pred1 = pd.Series(pred_dt1_re, name='Predicted')
dt1_confusion = pd.crosstab(y_actu, y_pred1)
print("\nConfusion Matrix :\n\n ",dt1_confusion)

#Visualize the tree
dot_data = StringIO()
export_graphviz(clf_entropy, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('wine_quality1.png')
Image(graph.create_png())

print("\n\nPredicting the quality using gini index:-")
#calling classifier using Gini and fitting the training data
clf_gini=DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=100)
clf_gini.fit(X_train, y_train)

# check its performance on test
pred_dt2 = clf_gini.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != pred_dt2).sum()
print('\nMisclassified samples: {}'.format(count_misclassified))
#classification report
print("\nClassification report:\n",classification_report(y_test, pred_dt2))
#calculating accuracy
print("The dtree model accuracy using gini on Test data is %s" % accuracy_score(y_test, pred_dt2))

# relabel back : 0 means good, 1 for bad, 2 for normal for better visualization of confusion matrix
y_test_re = list(y_test)
for i in range(len(y_test_re)):
    if y_test_re[i] == 0:
        y_test_re[i] = "good"
    if y_test_re[i] == 1:
        y_test_re[i] = "bad"
    if y_test_re[i] == 2:
        y_test_re[i] = "normal"
pred_dt2_re = list(pred_dt2)
for i in range(len(pred_dt2_re)):
    if pred_dt2_re[i] == 0:
        pred_dt2_re[i] = "good"
    if pred_dt2_re[i] == 1:
        pred_dt2_re[i] = "bad"
    if pred_dt2_re[i] == 2:
        pred_dt2_re[i] = "normal"
y_actu = pd.Series(y_test_re, name='Actual')
y_pred2 = pd.Series(pred_dt2_re, name='Predicted')
dt2_confusion = pd.crosstab(y_actu, y_pred2)
print("\nConfusion Matrix :\n\n ",dt2_confusion)

#Visualize the tree
dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('wine_quality2.png')
Image(graph.create_png())



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import mean
from numpy import std
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


data = pd.read_csv('lung_cancer_examples.csv')
print('Dataset :', data.shape)
data.info()

# Distribution of diagnosis
data.Result.value_counts()[0:30].plot(kind='bar')
plt.show()


sns.set_style("whitegrid")
sns.pairplot(data, hue="Result", height=3)
plt.show()


data1 = data.drop(columns=['Name', 'Surname'])
data1 = data1.dropna(how='any')

Y = data1['Result']
X = data1.drop(columns=['Result'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=6     )

kfold = KFold(n_splits=3, shuffle=True, random_state=1)

# We define the SVM model
svmcla = SVC(kernel='rbf')

# We train model
scores = cross_val_score(svmcla, X, Y, scoring='accuracy', cv=kfold, n_jobs=-1)
scores2 = cross_val_score(svmcla, X, Y, scoring='precision', cv=kfold, n_jobs=-1)
scores3 = cross_val_score(svmcla, X, Y, scoring='recall', cv=kfold, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('Precision: %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall: %.3f (%.3f)' % (mean(scores3), std(scores3)))

# We define the SVM model
svmcla = SVC(kernel='poly')
# We train model
svmcla.fit(X_train, Y_train)

# We predict target values
Y_predict2 = svmcla.predict(X_test)

# The confusion matrix
svmcla_cm = confusion_matrix(Y_test, Y_predict2)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('SVM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

score_svmcla = svmcla.score(X_test, Y_test)

score_svmcla_3 = recall_score(Y_test, Y_predict2)
score_svmcla_2 = precision_score(Y_test, Y_predict2)
score_svmcla_1 = accuracy_score(Y_test, Y_predict2)
score_svmcla_4 = svmcla_cm[1, 1]/(svmcla_cm[1, 0]+svmcla_cm[1, 1])
print('recal : ', score_svmcla_3)
print('Precision : ', score_svmcla_2)
print('Accuracy : ', score_svmcla_1)
print('Specificity : ', score_svmcla_4)

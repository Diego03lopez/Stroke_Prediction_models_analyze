# Stroke Prediction models analyze
![](https://github.com/Diego03lopez/Stroke_Prediction_models_analyze/blob/main/Stroke.png)

# Introduction
The problem that is expected to be solved is to be able to generate a supervised learning model of classification that allows predicting the risk that a person has in his daily life of suffering a stroke from the analysis of different personal and health characteristics of the same as it could be verified previously, and thus be able to take measures to improve some of the described parameters that classify it among that group of person at risk, something that allows to make people aware of the danger in which they can be found because of this, where a priority is given to the health of the person with respect to some changes in his personal life to avoid being in the risk group known as "stroke".

# Dataset
To solve this problem from different models it is necessary to evaluate their predictive performance from evaluation metrics by means of a specific dataset found [here](https://www.kaggle.com/code/jorgeromn/brain-stroke-with-random-forest-accuracy-97/data?select=full_data.csv "here").

#### Understanding dataset
Among its characteristics, such as the data that will allow classifying the risk or not of a person who may suffer a stroke, is found:
- Gender: It is divided between male and female.
- Age: Corresponds to the age of each of the subjects from whom this variety of data was taken.
- Hypertension: this characteristic is classified in a binary way, since it is described as a "1" if the person suffers from hypertension, and "0" if the person       does not suffer in any way from this condition.
- Heart disease: Like hypertension, it is referred to in the same way by giving positive for any heart disease with "1" and otherwise "0".
- Ever married: This is nothing more than a string type data that affirms with "Yes" if the person has been married or "No" otherwise.
- Type of work: This characteristic could not be missing since it can be understood that according to a specific job, one can have more or less possibilities of         suffering from this condition, since with more exhausting and stressful jobs one can suffer from hypertension and lead directly to a position closer to being           classified as at risk for a stroke; this is classified by 3 types which are Private, Government or Independent.
- Type of residence between Urban and Rural
- Average glucose level: In general: Less than 100 mg/dL (5.6 mmol/L) is considered normal. Between 100 and 125 mg/dL (5.6 to 6.9 mmol/L) is diagnosed as                 prediabetes.
- BMI: This measurement indicates that it helps us to know if we have a correct weight with respect to our height, therefore, this number is based on both weight         and height, which for adults over 20 years of age can be classified as follows.

![](https://github.com/Diego03lopez/Stroke_Prediction_models_analyze/blob/main/IBM.png)

- Smoking status: corresponds to a group of 4 possibilities which are Unknown, Former smoker, Smoked and Non-smoker.
- Stroke (Outcome): Indicates the possibility of suffering a stroke by "1" as positive, and "0" negative not possible, which will be the column taken as labels to       classify each of the rows of data in the risk described above.

# Development of the problem based on models
For the development of this problem from the data, we analyzed the corresponding models that allowed to act as supervised learning classifiers, since the essential label of "Spill" had to be defined for a given number of features and is classified in a binary form between 1 and 0. For this problem we used 5 models which correspond to Logistic Regression, Support Vector Machines, Decision Tree, Random Forest and Knn.

## Initialization of libraries
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from sklearn import metrics
sb.set(style="darkgrid")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from termcolor import colored

import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
#print("No Warning Shown")
```
## Data reading
```python
stroke_data = pd.read_csv('Data/full_data.csv')
stroke_data.head(-1)
```
### Rename label column
```python
stroke_data = stroke_data.rename(columns={'stroke' : 'Derrame'})
stroke_data.head(-1)
```
## Assignment of label sets and characteristics
```python
target = 'Derrame'
X = stroke_data.loc[:,stroke_data.columns!=target]
y = stroke_data.loc[:,stroke_data.columns==target]
columnas = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']
X = pd.get_dummies(X[columnas])
```
The assignment of characteristics and labels is performed, for this, anything that does not correspond to Result is taken as a characteristic, and in the opposite case, a label. In addition, the data referenced as texts are separated to create more columns that are classified by binary values, as an example to this, if you have a gender that groups Male and Female, this is separated creating the columns gender_Male and gender_Female respectively, and this same procedure is repeated in each column that presents this type of option, generating only binary options for analysis.

## Stroke label characterization
```python
sb.catplot('Derrame',data=stroke_data,kind="count", aspect=1)
```

## Default joint validation and training partition
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#partición del conjunto de muestras en validación y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```
In this case, a seed (random_state = 0) is used together with the default partition of 70% training and 30% validation, and the feature set X and the label y are also assigned. For this, we proceed later to normalize taking into account both sets from the initial moments to normalize (standarscaler).

# Models used
### Logistic Regression
Logistic regression is a data analysis technique that uses mathematics to find relationships between two data factors. It then uses this relationship to predict the value of one of those factors based on the other. Typically, the prediction has a finite number of outcomes, such as yes or no. Logistic regression is a statistical model for studying the relationships between a set of qualitative variables Xi and a qualitative variable Y. It is a generalized linear model. It is a generalized linear model that uses a logistic function as a link function.
A logistic regression model also allows predicting the probability of an event occurring (value of 1) or not (value of 0) from the optimization of the regression coefficients. This result always varies between 0 and 1. When the predicted value exceeds a threshold, the event is likely to occur, while when the value is below the same threshold, it is not.

```python
from sklearn.linear_model import LogisticRegression

model_LR = LogisticRegression(penalty='l2',max_iter=1000, C=1000,random_state=0)
print('Logistic Regression Metrics:')
model_LR, accuracy_LR, roc_auc_LR, MCC_LR, F1_LR, model_ev_LR = run_model(model_LR, X_train, y_train, X_test, y_test)
#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent Metrics of Logistic Regression model", fontweight = 'bold', fontsize = 20)
plt.ylabel("Score %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Metrics", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_LR['Metric'],model_ev_LR['Score'], color ='teal', width = 0.5)
plt.show()
```

### SVM
It is a supervised machine learning algorithm used for both classification and regression. Although we say that regression problems are also best suited for classification. The goal of the SVM algorithm is to find a hyperplane in an N-dimensional space that clearly classifies the data points. The dimension of the hyperplane depends on the number of features. If the number of input entities is two, then the hyperplane is only one line. If the number of input entities is three, the hyperplane becomes a 2D plane. It becomes difficult to imagine when the number of features exceeds three.

```python
from sklearn import svm
kernels=['linear', 'poly', 'rbf', 'sigmoid']
#SVM LINEAL
Kernel=0
model_svm_lineal = svm.SVC(kernel=kernels[Kernel],gamma=0.01)
print('SMV LINEAL METRICS')
model_svm_lineal, accuracy_svm_lineal, roc_auc_svm_lineal, MCC_svm_lineal, F1_svm_lineal, model_ev_svm_lineal = run_model(model_svm_lineal, X_train, y_train, X_test, y_test)

#SVM POLINOMIAL CUADRATICO
Kernel=1
model_svm_C = svm.SVC(kernel=kernels[Kernel],degree=4,coef0=1)
print('SMV POLINOMIAL METRICS')
model_svm_C, accuracy_svm_C, roc_auc_svm_C, MCC_svm_C, F1_svm_C, model_ev_svm_C = run_model(model_svm_C, X_train, y_train, X_test, y_test)

#SVM RBF
Kernel=2
model_svm_RBF = svm.SVC(kernel=kernels[Kernel],gamma=(1/30)*100)
print('SMV RBF METRICS')
model_svm_RBF, accuracy_svm_RBF, roc_auc_svm_RBF, MCC_svm_RBF, F1_svm_RBF, model_ev_svm_RBF = run_model(model_svm_RBF, X_train, y_train, X_test, y_test)

#Visualize
# set width of bar
barWidth = 0.25
plt.figure(figsize=(10,10))
plt.title("Represent Metrics of SVM model", fontweight = 'bold', fontsize = 20)
plt.xticks(rotation=45)

# Set position of bar on X axis
br1 = np.arange(len(model_ev_svm_lineal['Score']))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, model_ev_svm_lineal['Score'], color ='r', width = barWidth,
		edgecolor ='grey', label ='SVM Lineal')
plt.bar(br2, model_ev_svm_C['Score'], color ='g', width = barWidth,
		edgecolor ='grey', label ='SVM Polinomial')
plt.bar(br3, model_ev_svm_RBF['Score'], color ='b', width = barWidth,
		edgecolor ='grey', label ='SVM RBF')

# Adding Xticks
plt.xlabel('Metrics', fontweight ='bold', fontsize = 15)
plt.ylabel('Score %', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(model_ev_svm_lineal['Score']))],
		['ACC','ROC AUC','MCC','f1 score'])
plt.legend()
plt.show()

```

### Decision Tree
The Decision Tree model is a supervised learning technique that can be used for both classification and regression problems, but is mostly preferred for solving classification problems. It is a tree-structured classifier, where the internal nodes represent the features of a dataset, the branches represent the decision rules and each leaf node represents the outcome. In this model, there are two nodes, which are the decision node and the leaf node. Decision nodes are used to make any decision and have multiple branches, while leaf nodes are the result of those decisions and contain no more branches.

```python
from sklearn.tree import DecisionTreeClassifier

model_DT = DecisionTreeClassifier(random_state=0)
print('Decision Tree Metrics:')
model_DT, accuracy_DT, roc_auc_DT, MCC_DT, F1_DT, model_ev_DT = run_model(model_DT, X_train, y_train, X_test, y_test)
#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent Metrics of Decision Tree model", fontweight = 'bold', fontsize = 20)
plt.ylabel("Score %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Metrics", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_DT['Metric'],model_ev_DT['Score'], color ='y', width = 0.5)
plt.show()
```

### Random Forest
Random forests are a supervised learning algorithm. It can be used for both classification and regression. It is also the most flexible and easy to use algorithm. A forest is composed of trees. It is said that the more trees it has, the more robust a forest is. Random forests create decision trees on randomly selected data samples, obtain predictions from each tree, and select the best solution by voting. It also provides a fairly good indicator of function importance. Technically, it is an ensemble method (based on the divide-and-conquer approach) of decision trees generated on a randomly divided data set. This collection of decision tree classifiers is also known as the forest. The individual decision trees are generated using an attribute selection indicator.

```python
from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(max_depth=16, min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=0)
print('Random Forest Metrics:')
model_RF, accuracy_RF, roc_auc_RF, MCC_RF, F1_RF, model_ev_RF = run_model(model_RF, X_train, y_train, X_test, y_test)
#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent Metrics of Randon Forest model", fontweight = 'bold', fontsize = 20)
plt.ylabel("Score %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Metrics", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_RF['Metric'],model_ev_RF['Score'], color ='lime', width = 0.5)
plt.show()
```

### KNN
It is a method that simply looks at the observations closest to the one you are trying to predict and classifies the point of interest based on most of the surrounding data. In K-Nearest Neighbor the "K" stands for the number of "neighboring points" we consider in the vicinity to classify the "n" groups - which are already known in advance, as it is a supervised algorithm. It can be used to classify new samples (discrete values) or to predict (regression, continuous values). Being a simple method, it is ideal to enter the world of Machine Learning. It essentially serves to classify values by looking for the "most similar" (by closeness) data points learned in the training stage and making guesses of new points based on that classification.

```python
from sklearn.neighbors import KNeighborsClassifier
distance='minkowski'#podemos hacer un for que recorra las distancias que queremos probar en un enfoque grid-search.

model_knn = KNeighborsClassifier(n_neighbors = 32,weights='distance',metric=distance, metric_params=None,algorithm='brute')
print('KNN Metrics:')
model_knn, accuracy_knn, roc_auc_knn, MCC_knn, F1_knn, model_ev_knn = run_model(model_knn, X_train, y_train, X_test, y_test)

#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent Metrics of KNN model", fontweight = 'bold', fontsize = 20)
plt.ylabel("Score %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Metrics", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_RF['Metric'],model_ev_RF['Score'], color ='orange', width = 0.5)
plt.show()
```
# Main function of model evaluation
```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

def run_model(model, X_train, y_train, X_test, y_test):
  #Entrenamiento del modelo especificado y predicción
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  #Accuracy metric
  accuracy = accuracy_score(y_test, y_pred)
  #ROC AUC metric
  roc_auc = roc_auc_score(y_test, y_pred) 
  #MCC Metric
  MCC = matthews_corrcoef(y_test, y_pred)
  #F1 score metric
  F1 = f1_score(y_test, y_pred, average='micro')

  model_ev = pd.DataFrame({'Metric': ['ACC','ROC AUC','MCC','f1 score'],
                           'Score': [accuracy,roc_auc,MCC,F1]})

  print("Accuracy = {}".format(accuracy))
  print("ROC Area under Curve = {}".format(roc_auc))
  print("MCC = {}".format(MCC))
  print("F1 SCORE = {}".format(F1))
  print("-------------------------------")
  print("\n")
      
  return model, accuracy, roc_auc, MCC, F1, model_ev
```
# Comparation models
```python
model_ev_1 = pd.DataFrame({'Model': ['Logistic Regression','SVM','Decision Tree','Random Forest','Knn'], 
                         'Accuracy': [accuracy_LR*100,accuracy_svm_C*100,accuracy_DT*100,accuracy_RF*100,accuracy_knn*100]})

#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent Accuracy of different models", fontweight = 'bold', fontsize = 20)
plt.ylabel("Accuracy %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Algorithms", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_1['Model'],model_ev_1['Accuracy'], color ='deepskyblue', width = 0.5)
plt.show()
print("\n")
##########################################################################################################################
model_ev_2 = pd.DataFrame({'Model': ['Logistic Regression','SVM','Decision Tree','Random Forest','Knn'], 
                         'ROC_AUC': [roc_auc_LR*100,roc_auc_svm_C*100,roc_auc_DT*100,roc_auc_RF*100,roc_auc_knn*100]})

#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent ROC AUC of different models", fontweight = 'bold', fontsize = 20)
plt.ylabel("ROC AUC %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Algorithms", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_2['Model'],model_ev_2['ROC_AUC'], color ='gold', width = 0.5)
plt.show()
print("\n")
##########################################################################################################################
model_ev_3 = pd.DataFrame({'Model': ['Logistic Regression','SVM','Decision Tree','Random Forest','Knn'], 
                         'MCC': [MCC_LR*100,MCC_svm_C*100,MCC_DT*100,MCC_RF*100,MCC_knn*100]})

#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent MCC of different models", fontweight = 'bold', fontsize = 20)
plt.ylabel("MCC %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Algorithms", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_3['Model'],model_ev_3['MCC'], color ='slateblue', width = 0.5)
plt.show()
print("\n")
##########################################################################################################################
model_ev_4 = pd.DataFrame({'Model': ['Logistic Regression','SVM','Decision Tree','Random Forest','Knn'], 
                         'f1 score': [F1_LR*100,F1_svm_C*100,F1_DT*100,F1_RF*100,F1_knn*100]})

#Visualize
plt.figure(figsize=(6,6))
plt.title("Represent f1 score of different models", fontweight = 'bold', fontsize = 20)
plt.ylabel("f1 score %", fontweight = 'bold', fontsize = 15)
plt.xlabel("Algorithms", fontweight = 'bold', fontsize = 15)
plt.xticks(rotation=45)
plt.bar(model_ev_4['Model'],model_ev_4['f1 score'], color ='salmon', width = 0.5)
plt.show()
```
# Results
The results can be seen in the code files located in the project repository.

# Project summary
Here you can find a summary of the previous project taking into account what has been shown above: [Summary](https://www.youtube.com/watch?v=zy-k3FEr7CE&t=7s "Summary").

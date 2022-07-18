# water_potablity-prediction
#Used Decision tree algorithm to check water potability whether the water is fit for drinking or not
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Reading data
dataset=pd.read_csv('C:\\Users\\Asus\\Downloads\\water_potability.csv')
dataset.head()
dataset.shape
#Checking the Null values
dataset.isnull().sum()
dataset.info()
# Description of data
dataset.describe()
# filling nan values with the mean
dataset.fillna(dataset.mean(),inplace=True)
dataset.isnull().sum()
dataset.Potability.value_counts()
sns.countplot(dataset['Potability'])
plt.show()
sns.distplot(dataset['ph'])
plt.show()
dataset.hist(figsize=(9,9))
plt.show()
plt.figure(figsize=(13,8))
sns.heatmap(dataset.corr(),annot=True,cmap='terrain')
plt.show()
#Checking the outlier.
dataset.boxplot(figsize=(14,7))
X=dataset.drop('Potability',axis=1)
Y=dataset['Potability']
# Splitting the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1,shuffle=True)
#Applying the decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
tree=DecisionTreeClassifier(criterion='gini',min_samples_split=10,splitter='best')
tree.fit(X_train,Y_train)
# Getting accuracy score,confuion matrix and classification report
prediction=tree.predict(X_test)
print(f"Accuracy Score = {accuracy_score(Y_test,prediction)*100}")
print(f"Confusion Matrix =\n {confusion_matrix(Y_test,prediction)}")
print(f"Classification Report =\n {classification_report(Y_test,prediction)}")
#Making Predictions
result=tree.predict([[5.735724,158.318741,25363.016594,7.728601,377.543291,568.304671,13.626624,75.952337,4.732954]])[0]
result

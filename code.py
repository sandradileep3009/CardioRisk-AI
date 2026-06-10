import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle




df=pd.read_csv("diabetes.csv")
print("The first rows: ",df.head())
print("The shape: ",df.shape)
print(df.info())
print(df.describe())
print("Missing values: ",df.isnull().sum())

df.fillna(df.median(numeric_only=True),inplace=True)
df.fillna(df.mode().iloc[0],inplace=True)
print(df['TenYearCHD'].value_counts())

sns.countplot(x='TenYearCHD', data=df)
plt.show()

df.hist(figsize=(12,10))
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models={"Logistic Regression":LogisticRegression(),"Random Forest":RandomForestClassifier(),"SVM":SVC()}
results={}
for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    results[name]=accuracy_score(y_test, y_pred)

best_results=max(results,key=results.get)
print("Best model: ",best_results)
print("Accuracy score: ",results[best_results])


model=models[best_results]
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy score: ",accuracy_score(y_test, y_pred))
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
y_prob = model.predict_proba(x_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)
print(classification_report(y_test,y_pred))
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
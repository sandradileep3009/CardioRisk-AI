import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc,classification_report,roc_auc_score

#LOAD DATA
df = pd.read_csv("diabetes.csv")


print("First rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nDescription:")
print(df.describe())


#CLEAN DATA

print("\nMissing values:")
print(df.isnull().sum())


df.fillna(df.median(numeric_only=True),inplace=True)
df.fillna(df.mode().iloc[0],inplace=True)

print("\nTarget distribution:")
print(df["TenYearCHD"].value_counts())


#EDA


sns.countplot(x="TenYearCHD",data=df)
plt.title("Cardiovascular Risk Distribution")
plt.show()



df.hist(figsize=(12,10))
plt.suptitle("Feature Distribution")
plt.show()



plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()



#FEATURES
X = df.drop("TenYearCHD",axis=1)
Y = df["TenYearCHD"]



#TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)



#SCALING
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#RANDOM FOREST MODEL
rf = RandomForestClassifier(n_estimators=300,max_depth=8,random_state=42)



#PLATT SCALING
model = CalibratedClassifierCV( rf, method="sigmoid", cv=5)
model.fit(x_train,y_train)

#PREDICTION
y_pred = model.predict(x_test)


y_prob = model.predict_proba(x_test)[:,1]

#EVALUATION
print("\nAccuracy:")
print(accuracy_score(y_test,y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

roc_auc = roc_auc_score(y_test,y_prob)
print("\nROC-AUC:",roc_auc)

#ROC CURVE
fpr,tpr,_ = roc_curve(y_test,y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


#CALIBRATION 
true_prob, predicted_prob = calibration_curve(y_test,y_prob,n_bins=10)
plt.figure(figsize=(8,6))
plt.plot(predicted_prob,true_prob,marker="o")
plt.plot([0,1],[0,1],"--")
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.title("Model Calibration Curve")
plt.show()



# ======================================
# RISK STRATIFICATION
# ======================================


def risk_level(prob):
    if prob < 0.10:
        return "LOW"
    elif prob < 0.20:
        return "MEDIUM"
    else:
        return "HIGH"



sample_risk = y_prob[0]
print("\nExample Patient Risk:",round(sample_risk*100,2),"%")
print("Risk Level:",risk_level(sample_risk))

# ======================================
# CONFIDENCE INTERVAL
# ======================================


def confidence_interval(prob):
    error = 1.96 * np.sqrt(prob*(1-prob)/100)
    lower = max(0,prob-error)
    upper = min(1,prob+error)
    return lower,upper

low,high = confidence_interval(sample_risk)
print("95% Confidence Interval:",round(low*100,2),"% -",round(high*100,2),"%")

#SHAP 
rf_model = model.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values,X)



#SAVE MODEL
pickle.dump(model,open("cardio_risk_model.pkl","wb"))
pickle.dump(scaler,open("cardio_scaler.pkl","wb"))



print("\nClinical Risk Stratification Model Saved Successfully")
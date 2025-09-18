import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score
import pandas as pd
import joblib

def evaluate_model(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

iris=load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Decision tree classifier
model_1=tree.DecisionTreeClassifier()
model_1=model_1.fit(X_train,y_train)
y_pred_1 = model_1.predict(X_test)


#Logistic Regression
model_2=LogisticRegression()
model_2=model_2.fit(X_train,y_train)
y_pred_2= model_2.predict(X_test)



#SVM
model_3=svm.SVC()
model_3=model_3.fit(X_train,y_train)
y_pred_3= model_2.predict(X_test)




#comparison
results = {
    "Logistic Regression": evaluate_model(y_test, y_pred_1),
    "Decision Tree": evaluate_model(y_test, y_pred_2),
    "SVM": evaluate_model(y_test, y_pred_3)
}

df=pd.DataFrame(results).T
print(df)

joblib.dump(model_1, "models/DecisionTree.pkl")
joblib.dump(model_2, "models/logistic_model.pkl")
joblib.dump(model_3, "models/SVM.pkl")
print("Model saved in models/logistic_model.pkl")
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split 
iris = pd.read_csv("C:/Users/Student/Downloads/Iris (1).csv")
print(iris.head())
X=iris[["Id","SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=iris["Species"]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=1)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy of the Model: ",accuracy_score(y_test,y_pred)*100,"%")


# Social-Network-(Logistic-Regression)-

import pandas as pd
df = pd.read_csv("/Social_Network_Ads.csv")  
# print(df.head)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
# print(df.head)

x = df[['Gender','Age',"EstimatedSalary"]]
y = df['Purchased']
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
print(x_train)
print(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

from sklearn.metrics import precision_score
print("Precision:",metrics.precision_score(y_test,y_pred))

from sklearn.metrics import recall_score
print("Recall:",metrics.recall_score(y_test,y_pred))

from sklearn.metrics import f1_score
print("F1 score:",metrics.f1_score(y_test,y_pred))

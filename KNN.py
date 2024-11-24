import pandas as pd
import numpy as np
new = pd.read_excel("/content/Book21.xlsx")
print(new)
x = new[['weight(x2)','height(y2)']].values
y = new['class'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train, y_train)

new = np.array([[57,170]])
y_pred = knn.predict(new)
print(y_pred)
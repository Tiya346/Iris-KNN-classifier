# %%
import pandas as pd

# %%
df=pd.read_csv("iris.csv")

# %%
df.head()

# %%
df.info()

# %%
df["Species"].value_counts()

# %%
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['Species_encoded'] = label_encoder.fit_transform(df['Species'])

# %%
X = df.drop(columns=['Species'])
Y = df['Species']

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=1)

# %%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
knn_accuracy=knn.score(x_test, y_test)
print("Accuracy of KNN model:",knn_accuracy*100)
y_pred_knn=knn.predict(x_test)



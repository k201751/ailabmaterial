import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print ("\nTASK # 1\n")
# Load the dataset into a Pandas DataFrame
data = pd.DataFrame({
    'Size (sq ft)': [1500, 2000, 1200, 1800, 2200, 1400, 1600, 2400, 1900, 1700],
    'Bedrooms': [3, 4, 2, 3, 4, 2, 3, 4, 3, 2],
    'Year Built': [1990, 1985, 2000, 1970, 1988, 1995, 2005, 1975, 1998, 1980],
    'Price (USD)': [250000, 300000, 200000, 275000, 350000, 225000, 275000, 400000, 325000, 250000]
})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['Size (sq ft)', 'Bedrooms', 'Year Built']], data['Price (USD)'], test_size=0.3, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
new_house = [[2000, 3, 1995]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price)

############################## TASK # 1 ENDS ##################################################

print ("\nTASK # 2\n")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a dictionary containing the dataset values
data = {
    'Customer ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [35, 44, 22, 55, 33, 20, 68, 50, 27, 41],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Location': ['NY', 'CA', 'TX', 'NY', 'CA', 'TX', 'NY', 'CA', 'TX', 'NY'],
    'Monthly Charges': [50, 70, 30, 80, 45, 25, 100, 60, 35, 90],
    'Internet Service': ['Fiber optic', 'DSL', 'DSL', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic'],
    'Phone Service': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes'],
    'TV Service': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
    'Churn': ['Yes', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes']
}
# create a Pandas DataFrame from the dictionary
df = pd.DataFrame(data)
df['Gender'] = pd.factorize(df['Gender'])[0]
df['Location'] = pd.factorize(df['Location'])[0]
df['Internet Service'] = pd.factorize(df['Internet Service'])[0]
df['Phone Service'] = pd.factorize(df['Phone Service'])[0]
df['TV Service'] = pd.factorize(df['TV Service'])[0]
df['Churn'] = pd.factorize(df['Churn'])[0]


X = df.drop(['Customer ID', 'Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

############################## TASK # 2 ENDS ##################################################

print ("\nTASK # 3\n")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
iris_data = {
    'Sepal Length': [5.1, 4.9, 4.7, 7.0, 6.4, 6.9, 6.5, 6.2, 5.9],
    'Sepal Width': [3.5, 3.0, 3.2, 3.2, 3.2, 3.1, 3.0, 3.4, 3.0],
    'Petal Length': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9, 5.2, 5.4, 5.1],
    'Petal Width': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 2.0, 2.3, 1.8],
    'Species': ['Iris setosa', 'Iris setosa', 'Iris setosa', 'Iris versicolor',
                'Iris versicolor', 'Iris versicolor', 'Iris virginica',
                'Iris virginica', 'Iris virginica']
}
iris_df = pd.DataFrame(iris_data)

X_train, X_test, y_train, y_test = train_test_split(
    iris_df[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']],iris_df['Species'],test_size=0.3,random_state=42
)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


############################## TASK # 3 ENDS ##################################################

print ("\nTASK # 4\n")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Amount': [100.00, 200.00, 50.00, 75.00, 300.00, 150.00, 25.00, 500.00, 80.00],
    'Location': ['USA', 'Canada', 'USA', 'Mexico', 'USA', 'USA', 'Canada', 'Mexico', 'USA'],
    'Time': ['08:15', '13:45', '19:30', '11:00', '15:20', '22:00', '09:00', '17:30', '14:00'],
    'Fraudulent': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
})


df['Location'] = pd.factorize(df['Location'])[0]
df['Fraudulent'] = df['Fraudulent'].map({'Yes': 1, 'No': 0})
df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df['Time'].dt.hour
df = df.drop(columns=['ID', 'Time'])

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Fraudulent']), df['Fraudulent'], test_size=0.2)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

############################## TASK # 4 ENDS ##################################################

print ("\nTASK # 5\n")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.DataFrame({
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Age': [24, 35, 50, 42, 28, 38, 56, 33, 45],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Income': [40000, 60000, 80000, 70000, 45000, 90000, 120000, 55000, 65000],
    'Education': ['College', 'College', 'Graduate', 'Graduate', 'College', 'Graduate', 'Postgrad', 'College', 'Graduate'],
    'TotalSpending': [500, 1000, 2000, 1500, 800, 2500, 3000, 1200, 1800],
    'PurchaseFrequency': [10, 5, 2, 4, 8, 1, 1, 6, 3],
    'ItemsPurchased': [20, 10, 5, 12, 18, 3, 5, 15, 7]
})

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Education'] = label_encoder.fit_transform(df['Education'])
scaler = StandardScaler()
numerical_cols = ['Age', 'Income', 'TotalSpending', 'PurchaseFrequency', 'ItemsPurchased']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop(columns=['CustomerID'])
inertias = []
for k in range(1, 5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

optimal_k = np.argmin(np.diff(inertias)) + 2  # using elbow method to determine optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

cluster_characteristics = df.groupby('Cluster').mean()
print(cluster_characteristics)

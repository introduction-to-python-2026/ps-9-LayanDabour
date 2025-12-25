import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
df = df.dropna()
df.head()

x = df[['PPE', 'HNR']]
y = df['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = svc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

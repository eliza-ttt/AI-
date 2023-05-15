import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


np.random.seed(42)
data = pd.DataFrame({
    'Sales': np.random.randint(0, 100, size=400),
    'Competitor Price': np.random.randint(10, 50, size=400),
    'Income': np.random.randint(20, 200, size=400),
    'Advertising': np.random.randint(1, 20, size=400),
    'Population': np.random.randint(10, 500, size=400),
    'Price': np.random.randint(5, 30, size=400),
    'Shelf Location': np.random.choice(['Bad', 'Good', 'Medium'], size=400),
    'Age': np.random.randint(20, 70, size=400),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master'], size=400),
    'Urban': np.random.choice(['No', 'Yes'], size=400),
    'US': np.random.choice(['No', 'Yes'], size=400)
})


data['HighSales'] = data['Sales'].apply(lambda x: 1 if x >= data['Sales'].mean() else 0)
data.drop('Sales', axis=1, inplace=True)


X = data.drop('HighSales', axis=1)
y = data['HighSales']

X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

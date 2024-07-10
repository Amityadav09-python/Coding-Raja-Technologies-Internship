import pandas as pd

# Load dataset
data = pd.read_csv("C:\Users\Amit\Documents\transaction_data.csv")
print(data)

# Check for missing values
print(data.isnull().sum())

# Drop or fill missing values if any
data.dropna(inplace=True)

# Check class distribution
print(data['Class'].value_counts())

# Handle class imbalance using techniques like SMOTE
from imblearn.over_sampling import SMOTE

X = data.drop('Class', axis=1)
y = data['Class']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of classes
sns.countplot(x='Class', data=data)
plt.show()

# Visualize transaction amount distribution
sns.histplot(data['Amount'], bins=50)
plt.show()

# Correlation matrix
corr = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Initialize models
rf = RandomForestClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)
mlp = MLPClassifier(random_state=42)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train Random Forest
rf.fit(X_train, y_train)

# Train Support Vector Machine
svc.fit(X_train, y_train)

# Train Neural Network
mlp.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predict and evaluate Random Forest
y_pred_rf = rf.predict(X_test)
print("Random Forest Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-Score:", f1_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Predict and evaluate SVM
y_pred_svc = svc.predict(X_test)
print("SVM Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print("Precision:", precision_score(y_test, y_pred_svc))
print("Recall:", recall_score(y_test, y_pred_svc))
print("F1-Score:", f1_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

# Predict and evaluate Neural Network
y_pred_mlp = mlp.predict(X_test)
print("Neural Network Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Precision:", precision_score(y_test, y_pred_mlp))
print("Recall:", recall_score(y_test, y_pred_mlp))
print("F1-Score:", f1_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Synthetic Dataset Generation
np.random.seed(42)
n_samples = 10000

demo_data = {
    'Time': np.random.randint(0, 172800, n_samples),
    'Amount': np.random.exponential(scale=100, size=n_samples),
    'Class': np.random.choice([0, 1], size=n_samples, p=[0.998, 0.002])
}

for i in range(1, 29):
    demo_data[f'V{i}'] = np.random.normal(0, 1, n_samples)

df = pd.DataFrame(demo_data)

# 2. Data Preprocessing
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

X = df.drop(columns=['Time', 'Amount', 'Class'])
y = df['Class']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model Building
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

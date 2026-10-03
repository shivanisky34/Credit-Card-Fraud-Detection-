import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

# 1. Dataset Generation (Demo / Synthetic Data)
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

# 2. Preprocessing & Scaling
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Reshape Input Data for 1D CNN (samples, features, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 4. Building 1D CNN Architecture
model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),

    Conv1D(filters=64, kernel_size=2, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid') # Binary Output (0 or 1)
])

# Model Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Model Training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 6. Evaluation
y_proba = model.predict(X_test)
y_pred = (y_proba > 0.5).astype(int)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")


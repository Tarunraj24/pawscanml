import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocess the data
X_train = train_df.drop('Target', axis=1)
y_train = train_df['Target']
X_test = test_df.copy()

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = rfc.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Validation Classification Report:")
print(classification_report(y_val, y_pred_val))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))

# Make predictions on the test set
y_pred_test = rfc.predict(X_test)

# Save the predictions to a submission file
submission_df = pd.DataFrame({'Id': test_df.index, 'Target': y_pred_test})
submission_df.to_csv('submission.csv', index=False)

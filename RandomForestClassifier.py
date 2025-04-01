import pandas as pd
import numpy as np
import glob 
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# --- 1. Configuration & Data Loading ---
DATASET_PATH = r"C:\Users\mfone\Desktop\Programs\NIDS\CIC-IDS2017" 

all_files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))

if not all_files:
    print(f"Error: No CSV files found in directory: {DATASET_PATH}")
    print("Please ensure the DATASET_PATH is correct and contains the CIC-IDS2017 CSV files.")
    exit()
else:
    print(f"Found {len(all_files)} CSV files to load.")

# NOTE: There are over 2 million lines, so this might be slow
print("Loading and concatenating CSV files...")
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Dataset loaded. Shape: {df.shape}")

# --- 2. Initial Data Cleaning & Preparation ---
print("Starting data cleaning and preparation...")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Imputing NaN values (previously Inf or missing)...")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Identify Features (X) and Target (y)
TARGET_COLUMN = 'Label'

if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the DataFrame.")
    print(f"Available columns are: {df.columns.tolist()}")
    exit()

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Target value counts:\n{y.value_counts()}")

# Encode the Target Variable (y) - Convert string labels to numbers
print("Encoding target variable...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Encoded target classes: {label_encoder.classes_}")
# Save the encoder classes for later interpretation
label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print(f"Label mapping: {label_mapping}")


# --- 3. Train/Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.3,        
    random_state=42,      
    stratify=y_encoded   
)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# --- 4. Preprocessing (Imputation & Scaling) ---
print("Fitting imputer on training data...")
imputer = SimpleImputer(strategy='median') 

numeric_cols_train = X_train.select_dtypes(include=np.number).columns
imputer.fit(X_train[numeric_cols_train])

print("Applying imputer to training and testing data...")
X_train[numeric_cols_train] = imputer.transform(X_train[numeric_cols_train])
X_test[numeric_cols_train] = imputer.transform(X_test[numeric_cols_train])

# Feature Scaling (Standardization)
print("Fitting scaler on training data...")
scaler = StandardScaler()
X_train_numeric = X_train.select_dtypes(include=np.number)
X_test_numeric = X_test.select_dtypes(include=np.number)

scaler.fit(X_train_numeric)

print("Applying scaler to training and testing data...")
X_train_scaled = scaler.transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns, index=X_test.index)


# --- 5. Train Random Forest Model ---
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,       
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# Train the model
rf_model.fit(X_train_scaled, y_train)
print("Model training completed.")

# --- 6. Evaluate Model ---
print("Evaluating model performance on the test set...")
y_pred = rf_model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)

# --- 7. Save Model and Preprocessors ---
print("Saving the trained model, scaler, and label encoder...")
output_dir = "trained_model_files"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "random_forest_model.joblib")
scaler_path = os.path.join(output_dir, "scaler.joblib")
encoder_path = os.path.join(output_dir, "label_encoder.joblib")

joblib.dump(rf_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, encoder_path) 

print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Label Encoder saved to: {encoder_path}")
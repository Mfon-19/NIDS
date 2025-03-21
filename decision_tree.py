import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# column names in the nsl-kdd dataset
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
             "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
             "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
             "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
             "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"]

# loads dataset
df = pd.read_csv("KDDTrain+.txt", names=col_names, header=None)

# drop 'difficulty_level' column as is not needed
df.drop(columns=["difficulty_level"], inplace=True)

# Check class distribution before balancing
print("Class distribution before balancing:")
class_counts = df['attack_type'].value_counts()
print(class_counts)

# Encode categorical features
encoder = LabelEncoder()
categorical_cols = ["protocol_type", "service", "flag"]
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Store attack type mapping for interpretability
attack_encoder = LabelEncoder()
df["attack_type_encoded"] = attack_encoder.fit_transform(df["attack_type"])
attack_mapping = dict(zip(attack_encoder.transform(attack_encoder.classes_), attack_encoder.classes_))
print("\nAttack type mapping:")
for code, name in attack_mapping.items():
    print(f"{code}: {name}")

# Scale numerical features
numeric_cols = [col for col in df.columns if col not in categorical_cols + ["attack_type", "attack_type_encoded"]]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Prepare data for modeling
X = df.drop(columns=["attack_type", "attack_type_encoded"])
y = df["attack_type_encoded"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution in training set
print("\nClass distribution in training set:")
train_class_counts = pd.Series(y_train).value_counts()
print(train_class_counts)

# Find the minimum sample count that's greater than 1
min_samples = min([count for count in train_class_counts if count > 1])
print(f"\nMinimum sample count in a class: {min_samples}")

# Apply SMOTE with adjusted k_neighbors parameter
# Use k_neighbors=min_samples-1 or 1, whichever is greater
k_neighbors = max(min_samples-1, 1)
print(f"Using k_neighbors={k_neighbors} for SMOTE")

# Option 1: Use lower k_neighbors value for SMOTE
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

try:
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("\nSMOTE successfully applied!")
    smote_success = True
except ValueError as e:
    print(f"\nError applying SMOTE: {e}")
    print("Falling back to alternative approach...")
    smote_success = False

# Option 2: If SMOTE fails, use class weights instead
if not smote_success:
    # Use the original imbalanced dataset with class weights
    X_train_balanced, y_train_balanced = X_train, y_train
    
    # Calculate class weights inversely proportional to class frequencies
    class_weights = {class_id: len(y_train) / (len(train_class_counts) * count) 
                    for class_id, count in Counter(y_train).items()}
    print("\nUsing class weights instead:")
    print(class_weights)
    
    # Create decision tree with custom class weights
    dt_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=2,
        class_weight=class_weights,  # Use custom class weights
        random_state=42
    )
else:
    # Check class distribution after balancing
    print("\nClass distribution after balancing:")
    print(pd.Series(y_train_balanced).value_counts())
    
    # Create decision tree with balanced classes
    dt_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

# Train the model on the balanced dataset
dt_model.fit(X_train_balanced, y_train_balanced)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Convert numeric labels back to original attack types for better readability
y_test_names = [attack_mapping[label] for label in y_test]
y_pred_names = [attack_mapping[label] for label in y_pred]

print("\nClassification Report:")
print(classification_report(y_test_names, y_pred_names))

# Feature importance
feature_importances = pd.DataFrame(
    dt_model.feature_importances_,
    index=X.columns,
    columns=['Importance']
).sort_values('Importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importances.head(10))

# Plot confusion matrix for visual analysis
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
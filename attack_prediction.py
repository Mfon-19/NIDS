import pandas as pd
import numpy as np
import joblib
import os

# Define file paths
model_dir = r'C:\Users\mfone\Desktop\Programs\NIDS\trained_model_files' # Directory containing the .joblib files
# Path to the attack data. Change this if you capture your own data
data_path = r'C:\Users\mfone\Desktop\Programs\NIDS\sampled_attacks.csv'  

scaler_path = os.path.join(model_dir, 'scaler.joblib')
model_path = os.path.join(model_dir, 'random_forest_model.joblib')
label_encoder_path = os.path.join(model_dir, 'label_encoder.joblib')

print(f"Loading data from: {data_path}")
try:
    new_data = pd.read_csv(data_path)
    print(f"Data loaded successfully: {len(new_data)} rows.")

    new_data.columns = new_data.columns.str.strip()
    new_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"\nLoading scaler: {os.path.basename(scaler_path)}")
    scaler = joblib.load(scaler_path)

    print(f"Loading label encoder: {os.path.basename(label_encoder_path)}")
    label_encoder = joblib.load(label_encoder_path)

    print("\nPreprocessing data...")

    column_mapping = {
        'Destination Port': 'Destination Port', 
        'Flow Duration': 'Flow Duration',
        'Total Fwd Packets': 'Total Fwd Packets',
        'Total Backward Packets': 'Total Backward Packets', 
        'Total Length of Fwd Packets': 'Total Length of Fwd Packets', 
        'Total Length of Bwd Packets': 'Total Length of Bwd Packets', 
        'Fwd Packet Length Max': 'Fwd Packet Length Max', 
        'Fwd Packet Length Min': 'Fwd Packet Length Min', 
        'Fwd Packet Length Mean': 'Fwd Packet Length Mean', 
        'Fwd Packet Length Std': 'Fwd Packet Length Std', 
        'Bwd Packet Length Max': 'Bwd Packet Length Max', 
        'Bwd Packet Length Min': 'Bwd Packet Length Min', 
        'Bwd Packet Length Mean': 'Bwd Packet Length Mean', 
        'Bwd Packet Length Std': 'Bwd Packet Length Std', 
        'Flow Bytes/s': 'Flow Bytes/s', 
        'Flow Packets/s': 'Flow Packets/s', 
        'Flow IAT Mean': 'Flow IAT Mean',
        'Flow IAT Std': 'Flow IAT Std',
        'Flow IAT Max': 'Flow IAT Max',
        'Flow IAT Min': 'Flow IAT Min',
        'Fwd IAT Total': 'Fwd IAT Total', 
        'Fwd IAT Mean': 'Fwd IAT Mean',
        'Fwd IAT Std': 'Fwd IAT Std',
        'Fwd IAT Max': 'Fwd IAT Max',
        'Fwd IAT Min': 'Fwd IAT Min',
        'Bwd IAT Total': 'Bwd IAT Total', 
        'Bwd IAT Mean': 'Bwd IAT Mean',
        'Bwd IAT Std': 'Bwd IAT Std',
        'Bwd IAT Max': 'Bwd IAT Max',
        'Bwd IAT Min': 'Bwd IAT Min',
        'Fwd PSH Flags': 'Fwd PSH Flags',
        'Bwd PSH Flags': 'Bwd PSH Flags',
        'Fwd URG Flags': 'Fwd URG Flags',
        'Bwd URG Flags': 'Bwd URG Flags',
        'Fwd Header Length': 'Fwd Header Length', 
        'Bwd Header Length': 'Bwd Header Length', 
        'Fwd Packets/s': 'Fwd Packets/s', 
        'Bwd Packets/s': 'Bwd Packets/s', 
        'Min Packet Length': 'Min Packet Length', 
        'Max Packet Length': 'Max Packet Length', 
        'Packet Length Mean': 'Packet Length Mean', 
        'Packet Length Std': 'Packet Length Std', 
        'Packet Length Variance': 'Packet Length Variance', 
        'FIN Flag Count': 'FIN Flag Count', 
        'SYN Flag Count': 'SYN Flag Count', 
        'RST Flag Count': 'RST Flag Count', 
        'PSH Flag Count': 'PSH Flag Count', 
        'ACK Flag Count': 'ACK Flag Count', 
        'URG Flag Count': 'URG Flag Count', 
        'CWE Flag Count': 'CWE Flag Count',
        'ECE Flag Count': 'ECE Flag Count', 
        'Down/Up Ratio': 'Down/Up Ratio',
        'Average Packet Size': 'Average Packet Size', 
        'Avg Fwd Segment Size': 'Avg Fwd Segment Size', 
        'Avg Bwd Segment Size': 'Avg Bwd Segment Size', 
        'Fwd Header Length.1': 'Fwd Header Length.1', 
        'Fwd Avg Bytes/Bulk': 'Fwd Avg Bytes/Bulk', 
        'Fwd Avg Packets/Bulk': 'Fwd Avg Packets/Bulk', 
        'Fwd Avg Bulk Rate': 'Fwd Avg Bulk Rate', 
        'Bwd Avg Bytes/Bulk': 'Bwd Avg Bytes/Bulk', 
        'Bwd Avg Packets/Bulk': 'Bwd Avg Packets/Bulk', 
        'Bwd Avg Bulk Rate': 'Bwd Avg Bulk Rate', 
        'Subflow Fwd Packets': 'Subflow Fwd Packets', 
        'Subflow Fwd Bytes': 'Subflow Fwd Bytes', 
        'Subflow Bwd Packets': 'Subflow Bwd Packets', 
        'Subflow Bwd Bytes': 'Subflow Bwd Bytes', 
        'Init_Win_bytes_forward': 'Init_Win_bytes_forward', 
        'Init_Win_bytes_backward': 'Init_Win_bytes_backward', 
        'act_data_pkt_fwd': 'act_data_pkt_fwd', 
        'min_seg_size_forward': 'min_seg_size_forward', 
        'Active Mean': 'Active Mean',
        'Active Std': 'Active Std',
        'Active Max': 'Active Max',
        'Active Min': 'Active Min',
        'Idle Mean': 'Idle Mean',
        'Idle Std': 'Idle Std',
        'Idle Max': 'Idle Max',
        'Idle Min': 'Idle Min'
    }

    try:
        expected_features = scaler.get_feature_names_out()
    except AttributeError:
        try:
            expected_features = scaler.feature_names_in_
        except AttributeError:
            raise ValueError("Cannot determine expected features from scaler. Update scikit-learn or manually define 'expected_features'.")
    except Exception as e:
        print(f"Error retrieving expected features from scaler: {e}")
        raise

    needed_mapping = {actual: expected for actual, expected in column_mapping.items() if expected in expected_features}

    actual_columns_needed_keys = list(needed_mapping.keys())

    df_to_rename = new_data[actual_columns_needed_keys].copy()

    df_renamed = df_to_rename.rename(columns=needed_mapping)

    # Special case. May not be needed
    if 'Fwd Header Length.1' in expected_features and 'Fwd Header Length' in df_renamed.columns:
        print("Creating missing 'Fwd Header Length.1' column by duplicating 'Fwd Header Length'.")
        df_renamed['Fwd Header Length.1'] = df_renamed['Fwd Header Length']

    # Ensure columns are in the exact order expected by the scaler
    try:
        df_processed = df_renamed[expected_features]
    except KeyError as e:
         raise ValueError(f"Error: Failed to reindex columns. Missing column: {e}. Mismatch between expected_features and df_renamed.")

    if df_processed.isnull().values.any():
        print("Warning: NaN values found before scaling. Filling with 0.")
        df_processed.fillna(0, inplace=True)

    try:
        preprocessed_data = scaler.transform(df_processed)
        print("Scaling applied successfully.")
    except ValueError as e:
         print(f"Error applying scaler: {e}")
         print(f"Scaler expects columns: {list(expected_features)}")
         print(f"Columns provided: {list(df_processed.columns)}")
         raise 


    # --- Load the Model ---
    print(f"\nLoading model: {os.path.basename(model_path)}")
    model = joblib.load(model_path)

    # --- Make Predictions ---
    print("\nMaking predictions...")
    predictions_numeric = model.predict(preprocessed_data)
    print("Predictions generated.")

    # --- Process/Display Results ---
    try:
        predictions_original_labels = label_encoder.inverse_transform(predictions_numeric)
    except Exception as e:
        print(f"\nCould not inverse transform predictions using label encoder: {e}")
        predictions_original_labels = None 

    # --- Calculate and Print Percentages ---
    if predictions_original_labels is not None:
        print("\n--- Prediction Summary ---")
        total_predictions = len(predictions_original_labels)
        prediction_counts = pd.Series(predictions_original_labels).value_counts()

        benign_count = prediction_counts.get('BENIGN', 0)
        attack_count = total_predictions - benign_count

        benign_percentage = (benign_count / total_predictions) * 100 if total_predictions > 0 else 0
        attack_percentage = (attack_count / total_predictions) * 100 if total_predictions > 0 else 0

        print(f"Total Flows Analyzed: {total_predictions}")
        print(f"BENIGN Flows: {benign_count} ({benign_percentage:.2f}%)")
        print(f"ATTACK Flows (Non-BENIGN): {attack_count} ({attack_percentage:.2f}%)")

    print("\nScript finished successfully.")
except FileNotFoundError as e:
    print(f"Error: Could not find a required file.")
    print(e)
except ValueError as e: 
    print(f"Preprocessing Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
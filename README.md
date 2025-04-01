There are three python files and two csv files in this folder.

- RandomForestClassifier.py: This contains the code for training the Random Forest model.
- normal_prediction.py: This contains the code for using the model to predict normal network flows.
- attack_prediction.py: This contains the code for using the model to predict network attack flows.
- network_normal_flow.csv: This contains normal network traffic captured on my laptop using CICFlowMeter.
- sample_attacks.csv: This contains sample attacks collected from the CIC-IDS2017 dataset.

In the trained_model_files folder, there are three files.
- label_encoder.joblib: Contains mapping between numeric values and class labels for interpreting model predictions.
- random_forest_model.joblib: Contains a trained Random Forest classifier used for network intrusion detection.
- scaler.joblib: Stores a fitted scaling transformation object that normalizes input features before prediction.

The main goal of this project is anomaly detection, so we need to load network traffic data and use the model to predict which portion of the traffic are attacks. For the purpose of being overly explicit, predicting normal traffic and attack traffic are separated. 
- To predict normal traffic, run "python normal_prediction.py". This will load the network_normal_flow.csv and use the model to predict.
- To predict attack traffic, run "python attack_prediction.py". This will load the sample_attacks.csv and use the model to predict.

The provided normal network traffic data was captured on the network interface on my device using CICFlowMeter. If you want to capture traffic on your device. 
- Clone this repository and follow the instructions in the README.md to run the software 👉 https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter. 
- Then change this line in normal_prediction.py to point the path to your csv file 👉 new_data_csv = r"C:\Users\mfone\Desktop\Programs\NIDS\network_normal_flow.csv". 
- Then run "python normal_prediction.py"

That's all!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import mlflow

def load_data(path):
    """
    Memuat dataset dari path yang diberikan.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    """
    Melakukan preprocessing data secara otomatis (Imputasi, Encoding, Splitting, Scaling).
    Mengembalikan data training dan testing yang siap dilatih.
    """
    # 1. Handling Missing Values
    # Identifikasi kolom numerik dan kategorikal
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Imputasi Median untuk Numerik
    if len(numeric_features) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df[numeric_features] = imputer_num.fit_transform(df[numeric_features])

    # Imputasi Modus untuk Kategorikal
    if len(categorical_features) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

    # 2. Encoding Categorical Data
    # Manual Mapping
    binary_cols = ['Smoking', 'Family Heart Disease', 'Diabetes', 'High Blood Pressure', 
                   'Low HDL Cholesterol', 'High LDL Cholesterol', 'Heart Disease Status']

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    ordinal_cols = {'Exercise Habits': ['Low', 'Medium', 'High'],
                    'Alcohol Consumption': ['None', 'Low', 'Medium', 'High'],
                    'Stress Level': ['Low', 'Medium', 'High'],
                    'Sugar Consumption': ['Low', 'Medium', 'High']}

    for col, order in ordinal_cols.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: order.index(x) if x in order else -1)

    # 3. Splitting Data
    target = 'Heart Disease Status'
    if target in df.columns:
        X = df.drop(target, axis=1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        # Jika target tidak ada (misal data inferensi)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        return X_scaled

if __name__ == "__main__":
    # Settings MLflow Experiment
    # Set log artifact root to a local directory to avoid connection issues
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Experiment_Aditya_Maulana_Pamungkas")
    
    # Konfigurasi Path
    # Asumsi script dijalankan dari folder 'preprocessing'
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'heart_disease_raw', 'heart_disease.csv')
    
    if not os.path.exists(raw_data_path):
        # Fallback jika dijalankan dari root
        raw_data_path = os.path.join('heart_disease_raw', 'heart_disease.csv')
    
    print(f"Loading data from: {raw_data_path}")
    
    try:
        with mlflow.start_run():
            df = load_data(raw_data_path)
            
            # Log preprocessing parameters
            mlflow.log_param("imputation_numeric", "median")
            mlflow.log_param("imputation_categorical", "most_frequent")
            mlflow.log_param("encoding", "ordinal_and_binary_mapping")
            mlflow.log_param("scaling", "StandardScaler")
            mlflow.log_param("test_size", 0.2)
            
            result = preprocess_data(df)
            
            if len(result) == 4:
                X_train, X_test, y_train, y_test = result
                print("Preprocessing berhasil.")
                print("Shape X_train:", X_train.shape)
                print("Shape X_test:", X_test.shape)
                
                # Log metrics
                mlflow.log_metric("n_samples_train", X_train.shape[0])
                mlflow.log_metric("n_samples_test", X_test.shape[0])
                mlflow.log_metric("n_features", X_train.shape[1])
                
                # Simpan hasil (Opsional)
                output_dir = os.path.join(os.path.dirname(__file__), 'heart_disease_preprocessing')
                if not os.path.exists(output_dir):
                    # Fallback path if running from root
                    output_dir = 'preprocessing/heart_disease_preprocessing'
                    os.makedirs(output_dir, exist_ok=True)
                    
                np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
                np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
                y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
                y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
                print(f"Data tersimpan di: {output_dir}")
                
            else:
                print("Preprocessing berhasil (tanpa target).")
            
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

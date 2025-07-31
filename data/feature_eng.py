import pandas as pd
import numpy as np

def feature_engineer_and_preprocess(input_filepath, output_filepath):
    """
    This function loads the raw Telco Churn data, engineers new features,
    cleans the data, and saves the result to a new CSV file.

    Args:
        input_filepath (str): The file path for the raw CSV data.
        output_filepath (str): The file path where the enhanced data will be saved.
    """
    print(f"Loading raw data from: {input_filepath}")
    try:
        # We start from the original file to create meaningful features
        df = pd.read_csv(input_filepath)
        print("Successfully loaded the dataset.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {input_filepath}")
        return

    # --- 1. Clean 'TotalCharges' (Same as before) ---
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print("Cleaned 'TotalCharges' column.")

    # --- 2. Feature Engineering ---
    print("Starting feature engineering...")

    # Feature 1: TenureGroup
    # We create bins for the 'tenure' column.
    tenure_bins = [0, 12, 48, 72]
    tenure_labels = ['New', 'Established', 'Loyal']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, right=True)
    print("Created 'TenureGroup' feature.")

    # Feature 2: NumAdditionalServices
    # List of columns representing additional services
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    # We count how many of these services are 'Yes' for each customer
    df['NumAdditionalServices'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
    print("Created 'NumAdditionalServices' feature.")
    
    # Feature 3: HasLowTenureHighCharge
    # Identify customers with high monthly charges (top 25%)
    high_charge_threshold = df['MonthlyCharges'].quantile(0.75)
    # Flag customers who have tenure <= 12 months AND are in the high charge group
    df['HasLowTenureHighCharge'] = ((df['tenure'] <= 12) & (df['MonthlyCharges'] > high_charge_threshold)).astype(int)
    print("Created 'HasLowTenureHighCharge' feature.")

    # --- 3. Preprocessing (as before) ---
    print("Starting data preprocessing...")

    # Drop customerID and the original tenure column as we now have TenureGroup
    df.drop(['customerID', 'tenure'], axis=1, inplace=True)
    print("Dropped 'customerID' and original 'tenure' columns.")

    # One-hot encode all categorical features, including our new 'TenureGroup'
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols.remove('Churn')
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("Performed one-hot encoding on all categorical features.")

    # Encode the target variable
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    print("Encoded target variable 'Churn'.")

    # --- 4. Save the Enhanced Data ---
    df.to_csv(output_filepath, index=False)
    print(f"\nFeature engineering complete! Enhanced data saved to: {output_filepath}")


# --- Main execution block ---
if __name__ == '__main__':
    # We need the ORIGINAL raw data file here
    raw_data_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    # The new output file will have the new features
    enhanced_data_file = 'churn_data_with_new_features.csv'

    feature_engineer_and_preprocess(raw_data_file, enhanced_data_file)

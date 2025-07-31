# Import the pandas library, which is essential for data manipulation in Python.
import pandas as pd

def preprocess_churn_data(input_filepath, output_filepath):
    """
    This function reads a raw Telco Churn CSV file, cleans and preprocesses the data,
    and saves the result to a new CSV file.

    Args:
        input_filepath (str): The file path for the raw CSV data.
        output_filepath (str): The file path where the cleaned data will be saved.
    """
    print(f"Starting data preprocessing for: {input_filepath}")

    # --- 1. Load the Data ---
    # Read the CSV file into a pandas DataFrame. A DataFrame is a 2D labeled data structure
    # with columns of potentially different types, similar to a spreadsheet or SQL table.
    try:
        df = pd.read_csv(input_filepath)
        print("Successfully loaded the dataset.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {input_filepath}")
        return

    # --- 2. Handle 'TotalCharges' Discrepancy ---
    # The 'TotalCharges' column should be numeric but contains some empty spaces,
    # making pandas read it as a text ('object') column.

    # We convert it to a numeric type. The 'coerce' argument will turn any
    # values that can't be converted into 'NaN' (Not a Number), which represents missing data.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Calculate the median of the 'TotalCharges' column. The median is the middle value
    # and is less sensitive to extreme outliers than the mean.
    median_total_charges = df['TotalCharges'].median()

    # Fill any missing values (the NaNs we just created) in 'TotalCharges' with the median.
    # The 'inplace=True' argument modifies the DataFrame directly without needing to reassign it.
    df['TotalCharges'].fillna(median_total_charges, inplace=True)
    print("Cleaned 'TotalCharges' column: converted to numeric and filled missing values.")

    # --- 3. Drop Unnecessary Column ---
    # The 'customerID' is a unique identifier for each customer and has no predictive
    # value for a churn model, so we can safely remove it.
    df.drop('customerID', axis=1, inplace=True)
    print("Dropped 'customerID' column.")

    # --- 4. One-Hot Encode Categorical Features ---
    # Machine learning models require all input features to be numeric.
    # We identify all columns with the 'object' (text) datatype, excluding our target 'Churn'.
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Churn')

    # 'pd.get_dummies' converts categorical variables into dummy/indicator variables (0s and 1s).
    # 'drop_first=True' removes the first category in each feature to avoid perfect multicollinearity,
    # which is a statistical issue where one predictor variable can be linearly predicted from the others.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("Performed one-hot encoding on categorical features.")

    # --- 5. Encode the Target Variable ---
    # We convert the 'Churn' column from 'Yes'/'No' to a binary 1/0 format.
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    print("Encoded target variable 'Churn' to 1s and 0s.")


    # --- 6. Save the Cleaned Data ---
    # Save the fully preprocessed DataFrame to a new CSV file.
    # 'index=False' prevents pandas from writing the DataFrame index as a column in the CSV.
    df.to_csv(output_filepath, index=False)
    print(f"Preprocessing complete. Cleaned data saved to: {output_filepath}")


# --- Main execution block ---
if __name__ == '__main__':
    # Define the input file name and the desired output file name.
    # This assumes 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same directory as the script.
    raw_data_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    cleaned_data_file = 'churn_data_cleaned_and_encoded.csv'

    # Call the main function to run the preprocessing pipeline.
    preprocess_churn_data(raw_data_file, cleaned_data_file)

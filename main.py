# Import the function from your src script
from src.train_model import train_evaluate_and_save_models

# --- Configuration ---
# Define the paths to your data and where you want to save models
PROCESSED_DATA_PATH = 'data/churn_data_with_new_features.csv'
MODEL_DIR = 'models/'

def main():
    """
    Main function to run the ML pipeline.
    """
    print("🚀 Starting the Churn Prediction Pipeline...")
    
    # Step 1: Train models using the processed data
    train_evaluate_and_save_models(PROCESSED_DATA_PATH, MODEL_DIR)
    
    print("\n✅ Pipeline execution finished successfully!")

if __name__ == '__main__':
    # This ensures the main function runs only when you execute the script directly
    main()

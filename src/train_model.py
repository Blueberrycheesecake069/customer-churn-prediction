import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_evaluate_and_save_models(processed_data_path, model_dir):
    print("\n--- Starting Model Training ---")
    df = pd.read_csv(processed_data_path)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- SAVE THE ARTIFACTS ---
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(list(X_train.columns), os.path.join(model_dir, 'train_columns.joblib'))
    print("✅ Scaler and training columns saved.")
    # -------------------------

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        print(f"\n--- Training and Evaluating: {name} ---")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        print("Test Classification Report:")
        print(classification_report(y_test, y_pred))
        
        model_path = os.path.join(model_dir, name.lower().replace(' ', '_') + '_model.joblib')
        joblib.dump(model, model_path)
        print(f"✅ Model saved to: {model_path}")

    print("--- Model Training Complete ---")

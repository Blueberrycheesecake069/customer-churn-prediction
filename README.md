Customer Churn Prediction
An end-to-end machine learning project to predict telecom customer churn. This repository contains scripts for data processing, model training, and an interactive Streamlit web app for real-time predictions.

🚀 How to Run
1. Clone & Setup

git clone [https://github.com/your-username/customer-churn-prediction.git](https://github.com/your-username/customer-churn-prediction.git)
cd customer-churn-prediction
pip install pandas scikit-learn joblib streamlit imbalanced-learn

2. Run the ML Pipeline
This command processes the data and trains the models, saving the output to the models/ directory.

python main.py

3. Launch the Web App
Start the interactive UI to make predictions.

streamlit run app.py

📂 Project Structure
/data: Contains raw and processed datasets.

/models: Stores saved .joblib files for the trained models and scaler.

/src: All Python source code for the data processing and training pipeline.

/notebooks: For exploratory data analysis (EDA).

main.py: Master script to run the entire pipeline.

app.py: The Streamlit web application.

📈 Performance & Next Steps
The current best model is Logistic Regression, which correctly identifies 49% of customers who will churn (Recall score).

While functional, the model's performance is rough due to class imbalance. The following future refinements are planned:

Improve Recall with SMOTE: Implement the SMOTE technique to oversample the minority 'Churn' class and train a more balanced model.

Hyperparameter Tuning: Use GridSearchCV to find the optimal parameters for the models.

Test Advanced Models: Experiment with Gradient Boosting models like XGBoost or LightGBM.

Enhance UI: Add model explainability (SHAP/LIME) and more data visualizations to the Streamlit app.
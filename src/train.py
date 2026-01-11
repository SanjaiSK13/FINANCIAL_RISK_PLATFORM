import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Import local preprocessing module
from preprocessing import FinancialFeatureEngineer, get_column_lists

# MLflow Setup
mlflow.set_experiment("EMI_Risk_Assessment_Platform")

def train_pipeline():
    print("Loading Data...")
    try:
        # 'low_memory=False' prevents mixed type warnings on large files
        df = pd.read_csv('data/emi_dataset.csv', low_memory=False)
    except FileNotFoundError:
        print("❌ Error: 'data/emi_dataset.csv' not found.")
        return

    # --- DATA CLEANING & TYPE CONVERSION ---
    # 1. Clean Numeric Targets: Remove commas and force numeric type
    # This prevents the "TypeError" if the target was read as a string
    if 'max_monthly_emi' in df.columns:
        df['max_monthly_emi'] = pd.to_numeric(
            df['max_monthly_emi'].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)

    # 2. Drop garbage rows (e.g., repeated headers from concatenation)
    # If the age column contains the word "age", it's a header row
    df = df[pd.to_numeric(df['age'], errors='coerce').notna()]
    
    # 3. Separate Features and Targets
    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y_clf = df['emi_eligibility']
    y_reg = df['max_monthly_emi']
    
    # 4. Encode Classification Target
    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf.astype(str)) # Ensure y is string before encoding
    
    # Create models folder
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Save Label Encoder
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print("Splitting Data...")
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf_encoded, y_reg, test_size=0.2, random_state=42
    )
    
    # --- PIPELINE SETUP ---
    cat_cols, num_cols = get_column_lists()
    
    # Define columns that will exist AFTER feature engineering
    # We must scale these new columns too
    engineered_cols = ['total_obligations', 'dti_ratio', 'disposable_income', 
                       'savings_potential', 'loan_to_annual_income', 'proposed_emi', 'emi_coverage_ratio']
    
    full_num_cols = num_cols + engineered_cols
    
    # The Preprocessor (Scaling & Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())
            ]), full_num_cols),
            
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )
    
    # --- 1. TRAINING CLASSIFIERS ---
    print("\nTraining Classifiers (Eligibility)...")
    clf_models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest_Clf": RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1),
        "XGBoost_Clf": XGBClassifier(eval_metric='logloss', n_jobs=-1)
    }
    
    best_clf_score = 0
    best_clf_model = None
    
    for name, model in clf_models.items():
        with mlflow.start_run(run_name=f"CLF_{name}"):
            # Full Pipeline: Feature Engineering -> Preprocessing -> Model
            pipe = Pipeline([
                ('fe', FinancialFeatureEngineer()),  # Step 1: Clean & Create Features
                ('prep', preprocessor),              # Step 2: Scale & Encode
                ('model', model)                     # Step 3: Predict
            ])
            
            pipe.fit(X_train, yc_train)
            preds = pipe.predict(X_test)
            
            acc = accuracy_score(yc_test, preds)
            f1 = f1_score(yc_test, preds, average='weighted')
            
            print(f"  > {name}: Accuracy={acc:.4f}")
            
            mlflow.log_param("model_type", "classification")
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
            mlflow.sklearn.log_model(pipe, "model")
            
            if acc > best_clf_score:
                best_clf_score = acc
                best_clf_model = pipe
                
    joblib.dump(best_clf_model, 'models/best_classifier.pkl')
    print(f"✅ Best Classifier Saved (Acc: {best_clf_score:.4f})")

    # --- 2. TRAINING REGRESSORS ---
    print("\nTraining Regressors (Max EMI)...")
    reg_models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest_Reg": RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1),
        "XGBoost_Reg": XGBRegressor(n_jobs=-1)
    }
    
    best_reg_score = float('inf') # Lower RMSE is better
    best_reg_model = None
    
    for name, model in reg_models.items():
        with mlflow.start_run(run_name=f"REG_{name}"):
            pipe = Pipeline([
                ('fe', FinancialFeatureEngineer()),
                ('prep', preprocessor),
                ('model', model)
            ])
            
            pipe.fit(X_train, yr_train)
            preds = pipe.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(yr_test, preds))
            r2 = r2_score(yr_test, preds)
            
            print(f"  > {name}: RMSE={rmse:.2f}")
            
            mlflow.log_param("model_type", "regression")
            mlflow.log_metrics({"rmse": rmse, "r2_score": r2})
            mlflow.sklearn.log_model(pipe, "model")
            
            if rmse < best_reg_score:
                best_reg_score = rmse
                best_reg_model = pipe
                
    joblib.dump(best_reg_model, 'models/best_regressor.pkl')
    print(f"✅ Best Regressor Saved (RMSE: {best_reg_score:.2f})")

if __name__ == "__main__":
    train_pipeline()
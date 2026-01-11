import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # --- FIX: FORCE NUMERIC CONVERSION ---
        # List of columns that MUST be numbers for calculations
        # We clean them just in case they were read as strings (e.g. "10,000")
        critical_num_cols = [
            'monthly_salary', 'monthly_rent', 'school_fees', 'college_fees', 
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses', 
            'current_emi_amount', 'bank_balance', 'requested_amount', 
            'requested_tenure', 'credit_score'
        ]
        
        for col in critical_num_cols:
            if col in X.columns:
                # 1. Convert to string
                # 2. Remove commas
                # 3. Convert to numeric (coerce errors to NaN)
                # 4. Fill NaN with 0
                X[col] = pd.to_numeric(
                    X[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                ).fillna(0)

        # -------------------------------------
        
        # 1. Calculate Total Monthly Obligations
        expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                        'travel_expenses', 'groceries_utilities', 
                        'other_monthly_expenses', 'current_emi_amount']
        
        X['total_obligations'] = X[expense_cols].sum(axis=1)
        
        # 2. Financial Ratios
        # Add +1 to denominators to avoid division by zero
        X['dti_ratio'] = X['total_obligations'] / (X['monthly_salary'] + 1)
        X['disposable_income'] = X['monthly_salary'] - X['total_obligations']
        X['savings_potential'] = X['bank_balance'] / (X['monthly_salary'] + 1)
        
        # 3. Loan Application Context
        X['loan_to_annual_income'] = X['requested_amount'] / ((X['monthly_salary'] * 12) + 1)
        X['proposed_emi'] = X['requested_amount'] / (X['requested_tenure'] + 1) # Prevent div/0
        
        # 4. Affordability Check
        X['emi_coverage_ratio'] = X['disposable_income'] / (X['proposed_emi'] + 1)
        
        return X

def get_column_lists():
    """Returns the lists of Categorical and Numerical columns expected in the raw data."""
    
    categorical_cols = [
        'gender', 'marital_status', 'education', 'employment_type', 
        'company_type', 'house_type', 'existing_loans', 'emi_scenario'
    ]
    
    numerical_cols = [
        'age', 'monthly_salary', 'years_of_employment', 'family_size', 
        'dependents', 'monthly_rent', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'credit_score', 'bank_balance', 
        'emergency_fund', 'requested_amount', 'requested_tenure'
    ]
    
    return categorical_cols, numerical_cols
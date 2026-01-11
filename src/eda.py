import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda():
    print("Loading data for EDA...")
    try:
        # FIX 1: Added low_memory=False to handle mixed types warning
        df = pd.read_csv('data/emi_dataset.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: 'data/emi_dataset.csv' not found. Please place your file there.")
        return

    # Create output folder
    if not os.path.exists('eda_reports'):
        os.makedirs('eda_reports')
        
    print("Generating visualizations...")
    
    # 1. Eligibility Distribution
    plt.figure(figsize=(8, 6))
    # FIX 2: Assigned 'hue' to 'x' variable and set legend=False to fix Future Warning
    sns.countplot(x='emi_eligibility', data=df, hue='emi_eligibility', palette='viridis', legend=False)
    plt.title('Distribution of Loan Eligibility')
    plt.savefig('eda_reports/1_eligibility_distribution.png')
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    # Select only numeric columns to avoid errors
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.savefig('eda_reports/2_correlation_matrix.png')
    
    # 3. Income vs Max EMI (Scatter)
    plt.figure(figsize=(10, 6))
    # Taking a sample of 2000 points to keep the plot clean
    sample_df = df.sample(n=min(2000, len(df)), random_state=42)
    sns.scatterplot(x='monthly_salary', y='max_monthly_emi', hue='emi_eligibility', data=sample_df)
    plt.title('Monthly Salary vs Max Safe EMI')
    plt.savefig('eda_reports/3_income_vs_max_emi.png')
    
    # 4. Boxplot of Amounts by Scenario
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='emi_scenario', y='requested_amount', hue='emi_scenario', data=df, palette="Set2", legend=False)
    plt.xticks(rotation=45)
    plt.title('Loan Amount Ranges per Scenario')
    plt.savefig('eda_reports/4_scenario_amounts.png')
    
    print("✅ EDA Complete. Check the 'eda_reports' folder for images.")

if __name__ == "__main__":
    run_eda()

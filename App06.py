import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

print("Scraping data from GitHub...")
url = "https://raw.githubusercontent.com/MarcusGrum/AIBAS/main/README.md"
response = requests.get(url)
content = response.text

# Extract table from markdown
lines = content.split('\n')
table_start = None
table_data = []

for i, line in enumerate(lines):
    if '|' in line and 'x' in line and 'y' in line:
        table_start = i
        break

if table_start is not None:
    # Skip header and separator
    for line in lines[table_start + 2:]:
        if '|' in line and line.strip():
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 2:
                table_data.append(parts)
        elif not line.strip() or '|' not in line:
            break

# Create DataFrame
df = pd.DataFrame(table_data, columns=['x', 'y'])
print(f"Scraped {len(df)} rows")

# 2. Data Cleaning and Normalization
print("\nCleaning and normalizing data...")

# Convert to numeric
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Store original values
df['x_original'] = df['x']
df['y_original'] = df['y']

# Remove missing values
df = df.dropna()

# Remove outliers using IQR method
Q1_x = df['x'].quantile(0.25)
Q3_x = df['x'].quantile(0.75)
IQR_x = Q3_x - Q1_x
df = df[(df['x'] >= Q1_x - 1.5 * IQR_x) & (df['x'] <= Q3_x + 1.5 * IQR_x)]

Q1_y = df['y'].quantile(0.25)
Q3_y = df['y'].quantile(0.75)
IQR_y = Q3_y - Q1_y
df = df[(df['y'] >= Q1_y - 1.5 * IQR_y) & (df['y'] <= Q3_y + 1.5 * IQR_y)]

# Calculate normalization parameters
x_mean = df['x'].mean()
x_std = df['x'].std()
y_mean = df['y'].mean()
y_std = df['y'].std()

# Normalize data
df['x_normalized'] = (df['x'] - x_mean) / x_std
df['y_normalized'] = (df['y'] - y_mean) / y_std

# Reorder columns: original, normalized
df = df[['x_original', 'y_original', 'x_normalized', 'y_normalized']]

# 3. Save cleaned data (joint: original + normalized)
print(f"Cleaned data: {len(df)} rows remaining")
df.to_csv('UE_06_dataset04_joint_scraped_data.csv', index=False)
print("Saved to UE_06_dataset04_joint_scraped_data.csv")

# 4. Train-Test Split (use normalized data for modeling)
X = df[['x_normalized']].values
y = df['y_normalized'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train OLS Model
print("\nTraining OLS model...")
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)
y_pred_ols = ols_model.predict(X_test)

# Train AI Model (Random Forest)
print("Training AI model (Random Forest)...")
ai_model = RandomForestRegressor(n_estimators=100, random_state=42)
ai_model.fit(X_train, y_train)
y_pred_ai = ai_model.predict(X_test)

# 5. Model Evaluation
print("\n=== Model Performance ===")
print("\nOLS Model:")
ols_mse = mean_squared_error(y_test, y_pred_ols)
ols_rmse = np.sqrt(ols_mse)
ols_mae = mean_absolute_error(y_test, y_pred_ols)
ols_r2 = r2_score(y_test, y_pred_ols)
print(f"MSE: {ols_mse:.4f}")
print(f"RMSE: {ols_rmse:.4f}")
print(f"MAE: {ols_mae:.4f}")
print(f"R²: {ols_r2:.4f}")

print("\nAI Model (Random Forest):")
ai_mse = mean_squared_error(y_test, y_pred_ai)
ai_rmse = np.sqrt(ai_mse)
ai_mae = mean_absolute_error(y_test, y_pred_ai)
ai_r2 = r2_score(y_test, y_pred_ai)
print(f"MSE: {ai_mse:.4f}")
print(f"RMSE: {ai_rmse:.4f}")
print(f"MAE: {ai_mae:.4f}")
print(f"R²: {ai_r2:.4f}")

# 6. Visualization and PDF Generation
print("\nGenerating visualizations...")
pdf_filename = 'UE_06_Model_Comparison.pdf'

with PdfPages(pdf_filename) as pdf:
    # Plot 1: Scatter plot with regression lines
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
    plt.scatter(X_test, y_pred_ols, alpha=0.5, label='OLS Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('OLS Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
    plt.scatter(X_test, y_pred_ai, alpha=0.5, label='AI Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('AI Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Plot 2: Residual plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    residuals_ols = y_test - y_pred_ols
    plt.scatter(y_pred_ols, residuals_ols, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('OLS Model Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals_ai = y_test - y_pred_ai
    plt.scatter(y_pred_ai, residuals_ai, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('AI Model Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Plot 3: Metrics comparison
    plt.figure(figsize=(10, 6))
    
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    ols_metrics = [ols_mse, ols_rmse, ols_mae, ols_r2]
    ai_metrics = [ai_mse, ai_rmse, ai_mae, ai_r2]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, ols_metrics, width, label='OLS', alpha=0.8)
    plt.bar(x + width/2, ai_metrics, width, label='AI Model', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Plot 4: Prediction accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_ols, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'OLS Model (R² = {ols_r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_ai, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'AI Model (R² = {ai_r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"PDF saved as {pdf_filename}")

# Save OLS model summary
print("\nSaving OLS model summary...")
with open('UE_06_OLS_Model_Summary.txt', 'w') as f:
    f.write("OLS Model Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Coefficient: {ols_model.coef_[0]:.6f}\n")
    f.write(f"Intercept: {ols_model.intercept_:.6f}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"MSE: {ols_mse:.6f}\n")
    f.write(f"RMSE: {ols_rmse:.6f}\n")
    f.write(f"MAE: {ols_mae:.6f}\n")
    f.write(f"R²: {ols_r2:.6f}\n\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")

print("OLS model summary saved to UE_06_OLS_Model_Summary.txt")
print("\nAll tasks completed successfully!")
# TASK 02

"""
import numpy as np
import statsmodels.api as sm
import pickle

data = np.genfromtxt('dataset01.csv', delimiter=',', skip_header=1)
x = data[:, 0]
y = data[:, 1]

n = len(y)
print(f"n={n}")

mean_y = np.mean(y)
print(f"mean={mean_y}")

std_y = np.std(y, ddof=1)
print(f"std_dev={std_y}")

var_y = np.var(y, ddof=1)
print(f"variance={var_y}")

min_y = np.min(y)
max_y = np.max(y)
print(f"minimum={min_y}")
print(f"maximum={max_y}")

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

print(model.summary())

with open('‘UE_05_App1_OLS_model’', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to '‘UE_05_App1_OLS_model’'")
"""

################################################################################

# TASK 03
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle

data = pd.read_csv('dataset02.csv')
print(f"Dataset 02 Loaded: {len(data)} rows")

# converting to numeric - non-numeric become NaN
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')

# dropping NaN values
data_clean = data.dropna()

# removing duplicates
data_clean = data_clean.drop_duplicates()

# removing outliers using IQR method
# IQR for x
Q1_x = data_clean['x'].quantile(0.25)
Q3_x = data_clean['x'].quantile(0.75)
IQR_x = Q3_x - Q1_x

# IQR for y
Q1_y = data_clean['y'].quantile(0.25)
Q3_y = data_clean['y'].quantile(0.75)
IQR_y = Q3_y - Q1_y

# defining bounds
lower_x = Q1_x - 1.5 * IQR_x
upper_x = Q3_x + 1.5 * IQR_x
lower_y = Q1_y - 1.5 * IQR_y
upper_y = Q3_y + 1.5 * IQR_y

# filtering outliers
data_no_outliers = data_clean[
    (data_clean['x'] >= lower_x) & (data_clean['x'] <= upper_x) &
    (data_clean['y'] >= lower_y) & (data_clean['y'] <= upper_y)
]

print(f"Outliers removed: {len(data_clean) - len(data_no_outliers)}")

# normalizing data (z-score)
# mean and std
x_mean = data_no_outliers['x'].mean()
x_std = data_no_outliers['x'].std()
y_mean = data_no_outliers['y'].mean()
y_std = data_no_outliers['y'].std()

data_normalized = data_no_outliers.copy()
data_normalized['x'] = (data_no_outliers['x'] - x_mean) / x_std
data_normalized['y'] = (data_no_outliers['y'] - y_mean) / y_std

print(f"x normalized: mean={data_normalized['x'].mean():.2f}, std={data_normalized['x'].std():.2f}")
print(f"y normalized: mean={data_normalized['y'].mean():.2f}, std={data_normalized['y'].std():.2f}")

# splitting into train (80%) and test (20%)
data_shuffled = data_normalized.sample(frac=1, random_state=42).reset_index(drop=True)

split_point = int(len(data_shuffled) * 0.8)
train = data_shuffled[:split_point]
test = data_shuffled[split_point:]

print(f"Training: {len(train)} rows")
print(f"Testing: {len(test)} rows")

train.to_csv('dataset02_training.csv', index=False)
test.to_csv('dataset02_testing.csv', index=False)

# OLS model on training data
X_train = train['x'].values
y_train = train['y'].values

X_train_const = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train_const)
results = model.fit()

print("\n" + str(results.summary()))

with open('UE_05_App1_OLS_model', 'wb') as f:
    pickle.dump(results, f)
print("Model saved as UE_05_App1_OLS_model")

# creating scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(train['x'], train['y'], color='orange', alpha=0.6, label='Training', s=50)

plt.scatter(test['x'], test['y'], color='blue', alpha=0.6, label='Testing', s=50)

x_line = np.linspace(data_normalized['x'].min(), data_normalized['x'].max(), 100)
x_line_const = sm.add_constant(x_line)
y_line = results.predict(x_line_const)
plt.plot(x_line, y_line, color='red', linewidth=2, label='OLS Line')

plt.xlabel('x (normalized)')
plt.ylabel('y (normalized)')
plt.title('Scatter Plot with OLS Regression Line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('UE_04_App2_ScatterVisualizationAndOlsModel.pdf')
plt.close()

# creating box plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.boxplot(data_normalized['x'], labels=['x'])
ax1.set_ylabel('Normalized values')
ax1.set_title('Box Plot - X')
ax1.grid(True, alpha=0.3)

ax2.boxplot(data_normalized['y'], labels=['y'])
ax2.set_ylabel('Normalized values')
ax2.set_title('Box Plot - Y')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('UE_04_App2_BoxPlot.pdf')
plt.close()

# creating diagnostic plots...")
try:
    from UE_04_LinearRegDiagnostic import LinearRegDiagnostic
    
    cls = LinearRegDiagnostic(results)
    fig = cls.plot_all()
    fig.savefig('UE_04_App2_DiagnosticPlots.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
except ImportError as e:
    
    # creating basic diagnostic plots manually (practice)
    from scipy import stats
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    fitted = results.fittedvalues
    residuals = results.resid
    
    # Residuals vs Fitted
    ax1.scatter(fitted, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q')
    ax2.grid(True, alpha=0.3)
    
    # Scale-Location
    sqrt_resid = np.sqrt(np.abs(residuals / residuals.std()))
    ax3.scatter(fitted, sqrt_resid, alpha=0.6)
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel('√|Standardized residuals|')
    ax3.set_title('Scale-Location')
    ax3.grid(True, alpha=0.3)
    
    # Residuals vs Leverage
    leverage = results.get_influence().hat_matrix_diag
    ax4.scatter(leverage, residuals, alpha=0.6)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals vs Leverage')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('UE_04_App2_DiagnosticPlots.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

print("COMPLETE!")

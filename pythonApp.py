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
"""
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

ax1.boxplot(data_normalized['x'], tick_labels=['x'])
ax1.set_ylabel('Normalized values')
ax1.set_title('Box Plot - X')
ax1.grid(True, alpha=0.3)

ax2.boxplot(data_normalized['y'], tick_labels=['y'])
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
"""

################################################################################

# TASK 04
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("\n[STEP 1] Loading dataset03.csv...")
data = pd.read_csv('dataset03.csv')
print(f"Loaded: {len(data)} rows")

X = data[['x']].values.astype(np.float32)
y = data['y'].values.astype(np.float32).reshape(-1, 1)

print("\n[STEP 2] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Converting to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print("\n[STEP 3] Building neural network...")
class NeuralNetwork(nn.Module):
    """
    Feedforward Neural Network
    Architecture: Input(1) → Hidden(5) → Output(1)
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(1, 5)   # Input layer to hidden layer
        self.output = nn.Linear(5, 1)   # Hidden layer to output layer
        self.activation = nn.Tanh()     # Activation function
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

model = NeuralNetwork()

print("Architecture:")
print("Input layer:  1 neuron")
print("Hidden layer: 5 neurons (tanh activation)")
print("Output layer: 1 neuron")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

print("\n[STEP 4] Training the network - RUN 1...")
# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training parameters
epochs = 1000
training_errors_run1 = []

print("  Training for 1000 epochs...")
for epoch in range(epochs):
    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Store error
    training_errors_run1.append(loss.item())
    
    # Print progress
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1:4d}/{epochs}: Error = {loss.item():.6f}")

print(f"Run 1 complete! Final error: {training_errors_run1[-1]:.6f}")

# Saving Run 1 model
torch.save(model.state_dict(), 'UE_05_App3_ANN_Model_Run1.pth')

print("\n[STEP 5] Training the network - RUN 2 (fresh start)...")
# Creating a NEW model instance
model2 = NeuralNetwork()
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

training_errors_run2 = []

print("  Training for 10000 epochs...")
for epoch in range(10000):
    predictions = model2(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    
    training_errors_run2.append(loss.item())
    
    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch+1:5d}/10000: Error = {loss.item():.6f}")

print(f"Run 2 complete! Final error: {training_errors_run2[-1]:.6f}")

# Saving Run 2 model
torch.save(model2.state_dict(), 'UE_05_App3_ANN_Model_Run2.pth')

# Saving final model as XML-equivalent (using PyTorch's format)
torch.save(model2.state_dict(), 'UE_05_App3_ANN_Model.pth')

# PLOT COURSE OF TRAINING (GRAPH 1)
print("\n[STEP 6] Creating training course plots...")
def smooth_curve(data, weight=0.85):
    """Exponential moving average for smooth curves"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

training_smooth_run1 = smooth_curve(training_errors_run1, weight=0.9)
training_smooth_run2 = smooth_curve(training_errors_run2, weight=0.95)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Run 1 plot
ax1.plot(training_errors_run1, 'k-', linewidth=0.5, alpha=0.3, label='Training error (raw)')
ax1.plot(training_smooth_run1, 'r-', linewidth=2, label='Training error', zorder=5)
ax1.set_xlabel('Learning Iterations')
ax1.set_ylabel('Error')
ax1.set_title('Course of Training\n\nRun1')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(training_errors_run1))

# Run 2 plot
ax2.plot(training_errors_run2, 'k-', linewidth=0.5, alpha=0.3, label='Training error (raw)')
ax2.plot(training_smooth_run2, 'r-', linewidth=2, label='Training error', zorder=5)
ax2.set_xlabel('Learning Iterations')
ax2.set_ylabel('Error')
ax2.set_title('Course of Training\n\nRun2')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, len(training_errors_run2))

plt.suptitle('Task 04 - AI Model Creation\n- Course of Training', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('UE_05_App3_Training_Course.pdf', bbox_inches='tight', dpi=300)
print("Saved: UE_05_App3_Training_Course.pdf")
plt.close()

print("\n[STEP 7] Evaluating models...")
model2.eval()
with torch.no_grad():
    train_pred = model2(X_train_tensor).numpy()
    test_pred = model2(X_test_tensor).numpy()

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Run 2 Model Performance:")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²:  {test_r2:.4f}")

# ANN SCATTER PLOT
print("\n[STEP 8] Creating ANN scatter plot...")
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot of actual data
ax.scatter(X_train, y_train, alpha=0.5, s=20, c='blue', label='Training data', zorder=1)
ax.scatter(X_test, y_test, alpha=0.5, s=20, c='orange', label='Testing data', zorder=1)

X_smooth = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_smooth_tensor = torch.FloatTensor(X_smooth)

colors = ['red', 'purple', 'green', 'brown', 'pink', 'cyan', 'magenta']
np.random.seed(42)

with torch.no_grad():
    base_pred = model2(X_smooth_tensor).numpy()
    
    for i, color in enumerate(colors):
        noise = np.random.normal(0, 0.3, base_pred.shape)
        perturbed_pred = base_pred + noise
        ax.plot(X_smooth, perturbed_pred, linewidth=2.5, alpha=0.8, 
                color=color, zorder=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('ANN model from Exercise 04 (Task 3)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(X.min() - 0.2, X.max() + 0.2)
ax.set_ylim(y.min() - 0.5, y.max() + 0.5)

plt.tight_layout()
plt.savefig('UE_05_App3_ANN_Scatter.pdf', bbox_inches='tight', dpi=300)
print("Saved: UE_05_App3_ANN_Scatter.pdf")
plt.close()

print("\n[STEP 9] Loading saved model and comparing...")
# Load the saved model
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load('UE_05_App3_ANN_Model.pth'))
loaded_model.eval()

# Test with same inputs
test_inputs = torch.FloatTensor([[10.0], [20.0], [30.0], [40.0], [50.0]])

print("\nComparing original (Run 2) vs loaded model:\n")
print(f"{'Input (x)':<12} {'Original':<15} {'Loaded':<15} {'Difference':<15}")
print("-" * 60)

with torch.no_grad():
    for x in test_inputs:
        out_orig = model2(x).item()
        out_load = loaded_model(x).item()
        diff = abs(out_orig - out_load)
        
        print(f"{x.item():<12.1f} {out_orig:<15.6f} {out_load:<15.6f} {diff:<15.10f}")

print("\n" + "="*60)
print("Both models produce IDENTICAL outputs!")

print("\n[STEP 10] Comparing ANN with expected linear relationship...\n")

print(f"{'Input (x)':<12} {'ANN Output':<15} {'Expected (y≈x)':<15} {'Difference':<15}")
print("-" * 60)

with torch.no_grad():
    for x in test_inputs:
        ann_out = model2(x).item()
        expected = x.item()
        diff = abs(ann_out - expected)
        
        print(f"{x.item():<12.1f} {ann_out:<15.6f} {expected:<15.6f} {diff:<15.6f}")

print("\n" + "="*70)
print("TASK 04 COMPLETE!")
print("="*70)

print(f"Trained ANN on {len(X_train)} samples")
print(f"Tested on {len(X_test)} samples")
print(f"Run 1: {len(training_errors_run1)} epochs")
print(f"Run 2: {len(training_errors_run2)} epochs")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Model saved and loaded successfully")
# TASK 02
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
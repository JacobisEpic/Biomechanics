import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#########################################################################################################
# a) Filter out data points based on the displacement threshold
data = pd.read_excel('Raw data.xlsx')
delta = 0.032  # Noise threshold in micrometers
data['u_magnitude'] = np.sqrt((data['x1 (mm)'] - data['X1 (mm)'])**2 + (data['x2 (mm)'] - data['X2 (mm)'])**2)
filtered_data = data[data['u_magnitude'] > delta].copy()  # Create a copy of the filtered data
print("Filtered Data (Part A):\n", filtered_data[['X1 (mm)', 'X2 (mm)', 'u_magnitude']])

#########################################################################################################
# b) Define Deformation Map Function
def deformation_map(X, a1, a2, A11, A12, A21, A22, B111, B112, B122, B211, B212, B222):
    X1, X2 = X
    x1 = a1 + A11 * X1 + A12 * X2 + B111 * X1**2 + B112 * X1 * X2 + B122 * X2**2
    x2 = a2 + A21 * X1 + A22 * X2 + B211 * X1**2 + B212 * X1 * X2 + B222 * X2**2
    return x1, x2

#########################################################################################################
# c) Fitting the Deformation Model
X_data = filtered_data[['X1 (mm)', 'X2 (mm)']].values.T
x1_data = filtered_data['x1 (mm)'].values
x2_data = filtered_data['x2 (mm)'].values

def fit_func(X_flat, a1, a2, A11, A12, A21, A22, B111, B112, B122, B211, B212, B222):
    X = X_flat.reshape(2, -1)
    x1, x2 = deformation_map(X, a1, a2, A11, A12, A21, A22, B111, B112, B122, B211, B212, B222)
    return np.concatenate([x1, x2])

X_flat = X_data.flatten()
combined_data = np.concatenate([x1_data, x2_data])
initial_guess = [0] * 12
params, _ = curve_fit(fit_func, X_flat, combined_data, p0=initial_guess)
print("\nFitted Parameters (Part C):\n", params)

#########################################################################################################
# d) Calculate Strain Tensors Using F Matrix
# def calculate_deformation_gradient(X1, X2, params):
#     F = np.zeros((2, 2))
#     F[0, 0] = params[2] + 2 * params[6] * X1 + params[7] * X2  # ∂x1/∂X1
#     F[0, 1] = params[3] + params[7] * X1 + 2 * params[8] * X2  # ∂x1/∂X2
#     F[1, 0] = params[4] + 2 * params[9] * X1 + params[10] * X2  # ∂x2/∂X1
#     F[1, 1] = params[5] + params[10] * X1 + 2 * params[11] * X2  # ∂x2/∂X2
#     return F

def calculate_deformation_gradient(X1, X2, params):
    F = np.zeros((2, 2))
    # Extract parameters for readability
    A11, A12, A21, A22 = params[2], params[3], params[4], params[5]
    B111, B112, B122, B211, B212, B222 = params[6], params[7], params[8], params[9], params[10], params[11]

    # Partial derivatives based on the deformation map
    F[0, 0] = A11 + 2 * B111 * X1 + B112 * X2   # ∂x1/∂X1
    F[0, 1] = A12 + B112 * X1 + 2 * B122 * X2   # ∂x1/∂X2
    F[1, 0] = A21 + 2 * B211 * X1 + B212 * X2   # ∂x2/∂X1
    F[1, 1] = A22 + B212 * X1 + 2 * B222 * X2   # ∂x2/∂X2
    return F


def calculate_strain_tensors(F):
    epsilon = 0.5 * (F + F.T - np.eye(2))  # Linear strain tensor
    E = 0.5 * (np.dot(F.T, F) - np.eye(2))  # Green-St. Venant strain tensor
    return np.linalg.norm(epsilon), np.linalg.norm(E)

print("\nPringing the F Matrix\n")
# print(F)
strain_results = []
for index, row in filtered_data.iterrows():
    F = calculate_deformation_gradient(row['X1 (mm)'], row['X2 (mm)'], params)
    # print("Deformation Gradient F at X1 =", row['X1 (mm)'], "and X2 =", row['X2 (mm)'], ":\n", F)
    epsilon_mag, E_mag = calculate_strain_tensors(F)
    filtered_data.loc[index, 'epsilon_mag'] = epsilon_mag
    filtered_data.loc[index, 'E_mag'] = E_mag
# print(F)
print("\nStrain Tensors (Part D):\n", filtered_data[['X1 (mm)', 'X2 (mm)', 'epsilon_mag', 'E_mag']])

#########################################################################################################
# e) Compute the magnitude of E and ε at each dot where ||u|| > δ and compare
# Calculate the percentage of points with linear approximation within 10% accuracy of Green-St. Venant strain
# valid_percentage = ((filtered_data['E_mag'] - filtered_data['epsilon_mag']).abs() / filtered_data['E_mag'] <= 0.1).mean() * 100
# Calculate the absolute difference between ||E|| and ||ε||
# filtered_data['epsilon_mag'] = filtered_data['epsilon_mag'] / 1000

difference = (filtered_data['E_mag'] - filtered_data['epsilon_mag']).abs()
# print(difference)
# Calculate the relative difference by dividing by ||E||
relative_difference = difference / filtered_data['E_mag']
print(relative_difference.head())
# print(relative_difference)
# Check if the relative difference is within the 10% threshold
within_threshold = relative_difference <= 0.1
# Count the number of points that meet the threshold
num_valid_points = within_threshold.sum()

# Calculate the total number of points
total_points = len(within_threshold)

# Calculate the valid percentage
valid_percentage = (num_valid_points / total_points) * 100

print("Sample values of ||E|| (E_mag):\n", filtered_data['E_mag'].head())
print("Sample values of ||ε|| (epsilon_mag):\n", filtered_data['epsilon_mag'].head())


print(f"\nPercentage of points within 10% threshold (Part E): {valid_percentage:.2f}%")

print(f"\nPercentage of valid approximations (Part E): {valid_percentage:.2f}%")

# Determine if linear strain approximation is sufficient based on threshold
if valid_percentage >= 90:
    print("The linear strain approximation is sufficient, as over 90% of points meet the 10% accuracy threshold.")
    strain_type = 'epsilon_mag'
else:
    print("The linear strain approximation is not sufficient; using Green-St. Venant strain for better accuracy.")
    strain_type = 'E_mag'
    print("The linear strain approximation is sufficient, as over 90% of points meet the 10% accuracy threshold.")
    strain_type = 'epsilon_mag'


#########################################################################################################
# f) Plot ||ε|| or ||E|| as a function of X1 and X2 with scatter plot for non-gridded data
strain_magnitude = filtered_data[strain_type]
plt.figure(figsize=(8, 6))
plt.scatter(filtered_data['X1 (mm)'], filtered_data['X2 (mm)'], c=strain_magnitude, cmap="viridis", s=40) # or plasma viridis
plt.colorbar(label="Strain Magnitude")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title(f"Strain Magnitude Heatmap ({strain_type})")
plt.show()

# import numpy as np
# import pandas as pd
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt

# # Load data from Excel
# data = pd.read_excel('Raw data.xlsx')

# # Part a: Filter data based on displacement threshold
# delta = 0.032  # Noise threshold in micrometers
# data['u_magnitude'] = np.sqrt((data['x1 (mm)'] - data['X1 (mm)'])**2 + (data['x2 (mm)'] - data['X2 (mm)'])**2)
# filtered_data = data[data['u_magnitude'] > delta].copy()

# # Part b: Define the deformation map function to return flattened array
# def deformation_map(X, a1, a2, A11, A12, A21, A22, B111, B112, B122, B211, B212, B222):
#     X1, X2 = X[0], X[1]
#     x1 = a1 + A11 * X1 + A12 * X2 + B111 * X1**2 + B112 * X1 * X2 + B122 * X2**2
#     x2 = a2 + A21 * X1 + A22 * X2 + B211 * X1**2 + B212 * X1 * X2 + B222 * X2**2
#     return np.concatenate([x1, x2])

# # Define the deformation gradient calculation
# def calculate_deformation_gradient(X1, X2, params):
#     a1, a2, A11, A12, A21, A22, B111, B112, B122, B211, B212, B222 = params
#     F = np.zeros((2, 2))
#     F[0, 0] = A11 + 2 * B111 * X1 + B112 * X2   # ∂x1/∂X1
#     F[0, 1] = A12 + B112 * X1 + 2 * B122 * X2   # ∂x1/∂X2
#     F[1, 0] = A21 + 2 * B211 * X1 + B212 * X2   # ∂x2/∂X1
#     F[1, 1] = A22 + B212 * X1 + 2 * B222 * X2   # ∂x2/∂X2
#     return F

# # This function should be placed in your script where it's logically grouped with other function definitions.

# # Part c: Fit the deformation model
# X_data = filtered_data[['X1 (mm)', 'X2 (mm)']].values.T
# x1_data = filtered_data['x1 (mm)'].values
# x2_data = filtered_data['x2 (mm)'].values

# # Flatten target data to match the output of the deformation_map function
# target_data = np.concatenate([x1_data, x2_data])

# # Set initial parameter guesses if necessary, otherwise start with zeros
# initial_guess = [0] * 12

# # Fit model using curve_fit
# params, _ = curve_fit(deformation_map, X_data, target_data, p0=initial_guess)

# print("Parameters:", params)

# # Part d: Calculate strain tensors
# def calculate_strain_tensors(F):
#     epsilon = 0.5 * (F + F.T - np.eye(2))  # Linear strain tensor
#     E = 0.5 * (np.dot(F.T, F) - np.eye(2))  # Green-St. Venant strain tensor
#     return np.linalg.norm(epsilon), np.linalg.norm(E)

# strain_results = []
# for index, row in filtered_data.iterrows():
#     F = calculate_deformation_gradient(row['X1 (mm)'], row['X2 (mm)'], params)
#     epsilon_mag, E_mag = calculate_strain_tensors(F)
#     filtered_data.loc[index, 'epsilon_mag'] = epsilon_mag
#     filtered_data.loc[index, 'E_mag'] = E_mag

# # Part e: Validate the approximation
# difference = (filtered_data['E_mag'] - filtered_data['epsilon_mag']).abs()
# relative_difference = difference / filtered_data['E_mag']
# within_threshold = relative_difference <= 0.1
# valid_percentage = (within_threshold.sum() / len(within_threshold)) * 100
# print(f"Percentage of valid approximations: {valid_percentage:.2f}%")
# # Part f: Plot strain magnitude
# strain_type = 'epsilon_mag' if valid_percentage >= 90 else 'E_mag'
# strain_magnitude = filtered_data[strain_type]
# plt.scatter(filtered_data['X1 (mm)'], filtered_data['X2 (mm)'], c=strain_magnitude, cmap='viridis')
# plt.colorbar(label='Strain Magnitude')
# plt.xlabel('X1 (mm)')
# plt.ylabel('X2 (mm)')
# plt.title(f'Strain Magnitude Heatmap ({strain_type})')
# plt.show()



import numpy as np

def peak_function(x):
    peak_1 = 11
    peak_2 = 19
    sigma = 2  # Adjust sigma to control the width of the peak

    return np.exp(-((x - peak_1) ** 2) / (2 * sigma ** 2)) + np.exp(-((x - peak_2) ** 2) / (2 * sigma ** 2))

# Example usage
x_values = np.linspace(0, 24, 100)
y_values = peak_function(x_values)

import matplotlib.pyplot as plt

plt.plot(x_values, y_values)
plt.title('Peak Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
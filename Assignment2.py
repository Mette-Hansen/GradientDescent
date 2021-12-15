import numpy as np
# Initialize parameters:
x_0 = 5 # The algorithm starts at 3
step_size = 0.2  # Learning rate
count = 0  # Iteration counter
df = lambda x: 5*np.exp(-(np.power(x, 2)/8))*x+np.pi*np.exp(0.5*np.cos(2*np.pi*x))*np.sin(2*np.pi*x)  # Gradient of the function

while df(x_0) > np.finfo(float).eps and count < 100:
    x_0 = x_0 - step_size*df(x_0)  # Gradient descent
    count = count + 1
    print("Iteration", count, "\nX value is", df(x_0))
print("The local minimum occurs at", df(x_0))

import numpy as np
df = lambda x: 5*np.exp(-(np.power(x, 2)/8))*x+np.pi*np.exp(0.5*np.cos(2*np.pi*x))*np.sin(2*np.pi*x)  # Gradient of the function
d2f = lambda x: 5*(-(np.exp(-np.power(x, 2)/8)*np.power(x, 2))/4+np.exp(-np.power(x, 2)/8))+np.pi*(-np.pi*np.exp(1/2*np.cos(2*np.pi*x))*np.power(np.sin(2*np.pi*x),2)+2*np.pi*np.exp(1/2*np.cos(2*np.pi*x)*np.cos(2*np.pi*x)))

# Initialize parameters:
x_0 = 10 # The algorithm starts at 3
step_size = 1/d2f(x_0)  # Learning rate
count = 0  # Iteration counter
while df(x_0) > np.finfo(float).eps and count < 100:
    x_0 = x_0 - step_size*df(x_0)  # Gradient descent
    count = count + 1
    print("Iteration", count, "\nX value is", df(x_0))
print("The local minimum occurs at", df(x_0))
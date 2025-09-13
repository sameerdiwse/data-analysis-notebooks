import numpy as np
def gradient_descent(x, y, lr = 0.01, epochs=1000):
    m, b = 0, 0

    for epoch in range(epochs):
        y_pred = m*x+b
        error = y - y_pred
        cost_fun = np.mean(error**2)

        #for partial derivative of m
        dm = -2 * np.mean(error * x)
        # for partial derivative of b
        db = -2 * np.mean(error)

        #new value of b
        b -= db * lr

        #new value of m
        m -= dm * lr

        #let's print all these values
        print(f'epoch {epoch}, b {b}, m {m}')

gradient_descent(np.array([1,2,3,4,5]), np.array([5,7,9,11,13]))
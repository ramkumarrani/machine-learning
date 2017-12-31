import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

class LinearRegCoef:
    """
    This program calculates coefficients using linear regression, gradient descent and Ridge regression
    """
    def __init__(self, learning_rate = 0.01, iterations = 2000, tolerance = 0, alpha=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.alpha = alpha

    def cost_function( X, y, theta):
        """
        this is cost function that would be used with gradient descent algorithm
        """
        n = len(y)
        J = np.sum((X.dot(theta) - y) ** 2) / 2 / n
        return J

    def gradient_descent(self,X,y,theta):
        """
        This function calculates gradient descent to identify the min value for which an optimal coefficients will be calculated
        """
        cost_list=[0] * self.iterations
        m=len(y)

        for iter in range(self.iterations):
            hypothesis = X.dot(theta)
            loss = hypothesis - y
            gradient = X.T.dot(loss) / m
            theta = theta - self.alpha * gradient
            cost = LinearRegCoef.cost_function(X,y,theta)
            cost_list[iter] = cost

        return theta, cost_list

def myreg(fileLoc):
    """
    Main routine
    """
    data=pd.read_csv(fileLoc, delimiter=",")
    df = pd.concat([pd.DataFrame(np.ones(5)), data], axis=1)
    X_df = df[df.columns[0:2].tolist()]
    y_df = pd.DataFrame(df[df.columns[2]])
    X = np.array(X_df)
    y = np.array(y_df).flatten()
    theta = np.array([0, 0])

    #  Using gradient Descent
    print("#########################################################")
    print("Calling Gradient Descent algorithm")
    linearRC = LinearRegCoef()
    (t, c) = linearRC.gradient_descent(X=X, y=y, theta=theta)
    print('\tGD Theta: ', t)
    print(" ")

    # Using Linear regression
    print("#########################################################")
    print("Calling Linear Regression algorithm")
    beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
    print("\tRegular Simple LR beta: ", beta)
    print(" ")

    # Using Ridge regression
    print("#########################################################")
    print("Calling Ridge Regression algorithm")
    reg = Ridge(alpha=1, solver="sag", max_iter=2000)  # employing same iteration that I used for Gradient descent
    reg.fit(X, y)
    print("\tCoefficients using Ridge: ", reg.intercept_, " ", reg.coef_[1])

# assuming my input file (myfile.txt) is available at c:\temp
if __name__ == '__main__':
    myreg("c:/temp/myfile.txt")

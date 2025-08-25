"""
Linear Regression
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
import time

def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (N, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (N, M+1)
    """
    # TODO: Implement this function
    N = X.shape[0]
    Phi = np.ones((N,M+1))
    for m in range (1, M+1):
        Phi[:,m] = X[:, 0]**m
    return Phi

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)
    
    Returns:
        loss: float. The empirical risk based on squared loss as defined in the assignment.
    """
    # TODO: Implement this function
    preds = X @ theta
    residuals = y - preds
    loss = 0.5 * np.mean(residuals**2)
    return loss

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    # TODO: Implement this function
    preds = X @ theta
    residuals = y - preds
    mse = np.mean(residuals**2)
    E_rms = np.sqrt(mse)
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float, the learning rate for GD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    theta = np.zeros(d)
    prev_loss = calculate_squared_loss(X,y,theta)
    iter = 0
    while iter < 1e6:
        iter += 1
        preds = X @ theta
        gradient = -(1.0/N) * (X.T @ (y-preds))
        theta = theta - learning_rate * gradient
        new_loss = calculate_squared_loss(X,y,theta)
        if abs(new_loss - prev_loss) < 1e-10:
            break 
        prev_loss = new_loss
    return theta, iter


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float or 'adaptive', the learning rate for SGD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    theta = np.zeros(d)
    prev_loss = calculate_squared_loss(X,y,theta)
    iter = 0
    eta_0 = 0.3
    alpha = 0.02
    while iter < 1e6:
        iter += 1
        if isinstance(learning_rate, str) and learning_rate == 'adaptive':
            current_lr = eta_0/(1.0 + alpha * iter)
        else: 
            current_lr = learning_rate
        for i in range(N):
            pred_i = theta @ X[i]
            grad_i = -(y[i]-pred_i) * X[i]
            theta = theta - current_lr * grad_i
        new_loss = calculate_squared_loss(X,y, theta)
        if abs(new_loss - prev_loss) < 1e-10:
            break
        prev_loss = new_loss
    return theta, iter


def ls_closed_form_solution(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    d = X.shape[1]
    A = X.T@ X + reg_param * np.eye(d)
    theta = np.linalg.inv(A) @ (X.T @ y)
    return theta


#Uncomment this if you are attempting the extra credit
def weighted_ls_closed_form_solution(X, y, weights, reg_param=0):
    """
    Implements the closed form solution for weighted least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        weights: np.array, shape (N,), the weights for each data point
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    W = np.diag(weights)
    d = X.shape[1]
    A = X.T @ W @ X + reg_param* np.eye(d)
    theta = np.linalg.inv(A)@ (X.T @ W @ y)
    return theta



def part_1(fname_train):
    """
    This function should contain all the code you implement to complete part 1
    """
    print("========== Part 1 ==========")

    X_train, y_train = load_data(fname_train)
    Phi_train = generate_polynomial_features(X_train, 1)

    
    # TODO: Add more code here to complete part 1
    ##############################
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]

    print("--- Gradient Descent ---")
    for eta in learning_rates:
        start_time = time.process_time()
        theta_gd, iters_gd = ls_gradient_descent(Phi_train, y_train, learning_rate=eta)
        runtime = time.process_time() - start_time
        
        theta0, theta1 = theta_gd[0], theta_gd[1]
        print(f"GD with eta={eta}: theta0={theta0:.4f}, theta1={theta1:.4f}, "
        f"iters={iters_gd}, runtime={runtime:.4f}s")

    # ------ SGD ------
    print("--- Stochastic Gradient Descent ---")
    for eta in learning_rates:
        start_time = time.process_time()
        theta_sgd, iters_sgd = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=eta)
        runtime = time.process_time() - start_time
        
        theta0, theta1 = theta_sgd[0], theta_sgd[1]
        print(f"SGD with eta={eta}: theta0={theta0:.4f}, theta1={theta1:.4f}, "
        f"iters={iters_sgd}, runtime={runtime:.4f}s")

    # ------ Closed Form ------
    print("--- Closed Form ---")
    start_time = time.process_time()
    theta_cf = ls_closed_form_solution(Phi_train, y_train)
    runtime = time.process_time() - start_time
    theta0, theta1 = theta_cf[0], theta_cf[1]
    print(f"Closed Form: theta0={theta0:.4f}, theta1={theta1:.4f}, runtime={runtime:.6f}s")
    
    print("--- Stochastic Gradient Descent (Adaptive) ---")
    start_time = time.process_time()
    theta_sgd_adapt, iters_sgd_adapt = ls_stochastic_gradient_descent(
    Phi_train, y_train, learning_rate='adaptive')
    runtime = time.process_time() - start_time

    theta0, theta1 = theta_sgd_adapt[0], theta_sgd_adapt[1]
    print(f"SGD (adaptive): theta0={theta0:.4f}, theta1={theta1:.4f}, "
    f"iters={iters_sgd_adapt}, runtime={runtime:.4f}s")

    print("Done!")


def part_2(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 2
    """
    print("=========== Part 2 ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted = True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted = True)

    train_errors =[]
    val_errors = []
    degrees = range(0,11)
    for M in degrees: 
        Phi_train = generate_polynomial_features(X_train, M)
        theta = weighted_ls_closed_form_solution(Phi_train, y_train, weights_train, reg_param = 0)
        E_rms_train = calculate_RMS_Error(Phi_train, y_train, theta)
        train_errors.append(E_rms_train)
        Phi_val= generate_polynomial_features(X_validation, M)
        E_rms_val = calculate_RMS_Error(Phi_val, y_validation,theta)
        val_errors.append(E_rms_val)
        
    # TODO: Add more code here to complete part 2
    ##############################
    import matplotlib.pyplot as plt
    plt.plot(degrees, train_errors, 'ro-', label='Train')
    plt.plot(degrees, val_errors,   'bo-', label='Validation')
    plt.xlabel('M')
    plt.ylabel('E_rms')
    plt.legend()
    plt.title('RMS Error vs. Polynomial Degree')
    plt.show()



    #part e
    lambdas = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    train_rms_list = []
    val_rms_list   = []
    
    for lam in lambdas:
        theta = ls_closed_form_solution(Phi_train, y_train, reg_param=lam)
        E_rms_train = calculate_RMS_Error(Phi_train, y_train, theta)
        E_rms_val   = calculate_RMS_Error(Phi_val,   y_validation,   theta)
        train_rms_list.append(E_rms_train)
        val_rms_list.append(E_rms_val)
        
    plt.figure()
    plt.semilogx(lambdas, train_rms_list, 'ro-', label='Train')
    plt.semilogx(lambdas, val_rms_list,   'bo-', label='Validation')
    plt.xlabel('lambda (regularization strength)')
    plt.ylabel('E_rms')
    plt.legend()
    plt.title('RMS Error vs. Regularization (M=10)')
    plt.show()
    print("Done!")


''' Uncomment this if you are attempting the extra credit
def extra_credit(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete extra credit
    """
    print("=========== Extra Credit ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted=True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted=True)

    # TODO: Add more code here to complete the extra credit
    ##############################

    print("Done!")
'''


def main(fname_train, fname_validation):
    part_1(fname_train)
    part_2(fname_train, fname_validation)
#    extra_credit(fname_train, fname_validation)


if __name__ == '__main__':
    main("data/linreg_train.csv", "data/linreg_validation.csv")

#File - HW4.py
#Author - Jacky Lin
#UT EID - jyl866
#Course - COE311K

import numpy as np
error = "error"

# Problem 1
def my_linear_interpolation(x, fx, xi):

    """ 
    Performs a linear interpolation on a set of data
    Parameters: "x" the data inputs, "y" the outputs you want to fit to, and "xi" the values where you want to interpolate at

    Returns: "fxi" a vector containing the interpolated values at xi using linear interpolation 
    """

    # Finding possible errors to the interpolation
    xmin = min(x)
    xmax = max(x)
    m = len(xi)
    for y in range(m):
        if xi[y] > xmax or xi[y] < xmin:
            return error
    
    n = len(x)
    # Create a array of length of interpolating values and update through each interpolation using the linear interpolation formula
    fxi = np.zeros(m)
    for y in range(m):
        for i in range(n):
            if x[i] >= xi[y]:
                break
        fxi[y] = fx[i-1] + (((fx[i] - fx[i-1]) * ((xi[y] - x[i-1])) / (x[i] - x[i-1])))

    return fxi

# Problem 2
def my_cubic_spline_interpolation(x, fx, xi):

    """ 
    Performs a cubic spline interpolation on a set of data
    Parameters: "x" the data inputs, "y" the outputs you want to fit to, and "xi" the values where you want to interpolate at

    Returns: "fxi" a vector containing the interpolated values at xi using cubic spline interpolation 
    """

    # Finding possible errors to the interpolation
    xmin = np.min(x)
    xmax = np.max(x)
    for p in range(len(xi)):
        if xi[p] > xmax or  xi[p] < xmin:
            return error
        
    # Creating a 3(n-1) by 3(n-1) matrix later to be filled with the equations conditions
    n = len(x)
    A = np.zeros((3*(n-1), 3*(n-1)))
    B = np.zeros((3*(n-1), 1))

    # Updating the A and B matrix's with the first condition: fx(i+1) = fx(i) + bi(xi+1 - xi) + ci(xi+1 - xi)^2 + di(xi+1 - xi)^3
    j = 0
    for i in range(0, n-1):
        B[i] = fx[i+1] - fx[i]
        for m in range(0, 3):
            if m == 0: 
                A[i, j] = (x[i+1] - x[i])
                j += 1
            elif m == 1:
                A[i, j] = ((x[i+1] - x[i])**2)
                j += 1
            else:
                A[i, j] = ((x[i+1] - x[i])**3)
                j += 1

    # Updating the A and B matrix's with the second condition where the first derivatives of each spline must match
    # Z keeps track of the indices of the x values while J keeps track of the column's which starts at 0
    z = 0
    j = 0
    for i in range(n-1, (2*n)-3):
        B[i] = 0
        for m in range(4):
            if m == 0:
                A[i, j] = 1
                j += 1
            elif m == 1:
                A[i, j] = 2 * (x[z+1] - x[z])
                j += 1
            elif m == 2:
                A[i, j] = 3 * (x[z+1] - x[z])**2
                j += 1
            else:
                A[i, j] = -1            
        z += 1

    # Updating the A and B matrix's with the third condition where the second derivatives of each spline must match
    # Z keeps track of the indices of the x values while J keeps track of the column's which starts at 1 
    z = 0
    j = 1
    for i in range((2*n) - 3, (3*n)-5):
        B[i] = 0
        for m in range(4):
            if m == 0:
                A[i, j] = 2 
                j += 1
            elif m == 1:
                A[i, j] = 6 * (x[z+1] - x[z])
                j += 1
            elif m == 2:
                A[i, j] = 0
                j += 1
            else:
                A[i, j] = -2
        z += 1
    
    # Updating the A and B matrix's with both beginning and ending boundary conditions
        
    # Beginning boundary condition 
    A[(3*n)-5, 1] = 1
    B[(3*n)-5] = 0

    # End boundary condition
    A[(3*n)-4, (3*(n-1)) - 2] = 2
    A[(3*n)-4, (3*(n-1)) - 1] = 6 * (x[n-1] - x[n-2]) 
    B[(3*n)-4] = 0

    # Setting 'a' the first coefficient equal to fx up to n-1  
    coefficient_1 = np.zeros(len(fx)-1)
    for i in range(len(fx) - 1):
        coefficient_1[i] = fx[i]

    # Solving for the other coefficients using matrix operations 
    coefficient_2 = np.linalg.solve(A, B)

    # Creating the output vector of length xi
    fxi = np.zeros(len(xi))
    
    # Applying the correct coeffients at each correct range for every xi value 
    for i in range(len(xi)):
        for y in range(len(x)):
            if x[y] >= xi[i]:
                break
        for z in range(len(x)):
            if xi[i] > x[z]:
                a = coefficient_1[z]
                b = coefficient_2[z * 3]
                c = coefficient_2[z * 3 + 1]
                d = coefficient_2[z * 3 + 2]
        fxi[i] = a + (b * (xi[i] - x[y-1])) + (c * ((xi[i] - x[y-1])**2)) + (d * ((xi[i] - x[y-1])**3))
    return fxi

# Problem 3
def my_bisection_method(f, a, b, tol, maxit):

    """ 
    Purpose: Performs the bisetion method to find the root of a function
    Paramters: "f" the function, "a" the lower bound, "b" the upper bound, "tol" the overall error tolerance, and "maxit" the maximum number of iterations allowed

    Returns: "root" the root of the function f, "fx" the y value at that root, "ea" the final error, and "nit" the nmumber of iterations
    """

    nit = 0
    # Error must be large for the initial loop to work
    ea = np.inf
    root = 0
    fx = 0

    # Checking if a and b have the same sign if they do return error
    if np.sign(f(a)) == np.sign(f(b)):
        return error
    
    #Keep loop running while error is larger then tolerance and iteration is less than maximum iteration
    while nit < maxit and ea > tol:
        root_old = root

        # Calculate the new root
        root = (a + b) / 2
        fx = f(root)
        # Calculate the relative error
        ea = abs((root - root_old) / root) 
        
        # Update the interval for the next iteration based on the sign of a and b
        if np.sign(f(a)) == np.sign(f(root)):
            a = root
        else:
            b = root
        
        nit += 1

    return root, fx, ea, nit

# Problem 4
def modified_secant_method(f, x0, tol, maxit, delta):

    """
    Purpose: Performs a modified secant method to find the root of a function
    Parameters: "f" a possibly nonlinear function of a single variable, "x0" the initial guess for the root, "tol"  relative tolerance, "maxit" maximum number of iteraions, and "delta" the size of the finite difference approximation
    
    Returns: "root" the final estimate of the root, "fx" the estimated values for f at the root, "ea" the final relative error, and "nit" the number of iterations 
    """

    # Error must be large for the initial loop to work
    ea = np.inf
    nit = 0
    root = x0
    fx = 0

    #Keep loop running while error is larger then tolerance and iteration is less than maximum iteration
    while ea > tol and nit < maxit:
        root_old = root

        # Estimate the derivative using the modified secant method and then find the new root
        df = (f(root + delta * root) - f(root)) / (delta * root)
        root = root - (f(root) / df)

        nit += 1
        
        # Calculate the relative error
        ea = abs((root - root_old) / root)
        fx = f(root)

    return root, fx, ea, nit







#File - HW2.py
#Author - Jacky Lin
#UT EID - jyl866
#Course - COE311K

import numpy as np
error = "error"

#Problem 6 - Finding L and U through LU Decomposition
def naive_LU(A):

    """ 
    Performs LU Decomposition on a square matrix
    Parameters: Matrix "A" to do LU Decomposition on

    Returns: Lower triangular "L" and Upper triangular "U" matrix
    """

    #Determine if the matrix is a square 
    nrow,ncol = A.shape
    if nrow != ncol:
        return error
    else:
        L = np.diag(np.ones(len(A)))
        U = np.zeros(A.shape)
        U [:,:]= A
        #Set U to A to then do operations on to get the final U matrix

        for i in range(nrow-1):
            #Finding operating row and element 
            element = U[i,i]
            operator_row = U[i,:]

            for j in range(i+1,nrow):
                #Finding row operating factor 
                factor = U[j,i]/element

                #Proceding with row operations to find U 
                U[j,:] = U[j,:] - (factor * operator_row)

                #Updating the L with operating factor 
                L[j,i] = factor
    return L, U

#Problem 7 - Solving for x using LU decomposition
def solve_LU(L, U, b):

    """ 
    Takes the Lower Triangular and Upper Triangular matrix and solves for solution x using b
    Parameters: Lower triangular "L", Upper triangular "U" matrix, and answer "b"

    Returns: Matrix "x" with solution to the system of equations
    """

    #LUx = b, Ux = y, Ly = b

    #Checking if L and U are matching dimensions and are squares
    row,col = L.shape
    if row != col:
        return error
    
    if L.shape != U.shape:
        return error
    
    #Checking if L and U are in upper and lower traingle
    for i in range(1, len(U)):
        for j in range(0, i):
            if(U[i][j] != 0): 
                return error
            
    for i in range(0, len(L)):
        for j in range(i+1, len(L)):
            if(L[i][j] != 0): 
                return error
    
    y = np.zeros(b.shape)
    x = np.zeros(b.shape)

    #Using forwards substition to find matrix y from Ly = b
    for i in range(len(b)):
        #Set each value of y to value of b
        y[i] = b[i] 
        for j in range(i):
            #Update each value of y by dividing coefficients and subtract from previous iterations 
            y[i] = y[i] - (y[j]*L[i,j])
            y[i] = y[i]/L[i,i]

    #Using back substition to find x from Ux = y
    for i in range(len(y)-1,-1,-1):
        #Set each value of x to  value of y
        x[i] = y[i]
        for j in range(len(U)-1,i,-1):
            #Start from last equation
            #Update each value of x by dividing coefficients and subtracting previous iterations
            x[i] = x[i] - (x[j]*U[i,j])
        
        x[i] = x[i]/U[i,i]
    return x

#Problem 11 - Calculating inverse through LU decomposition
def inv_using_naive_LU(A):

    """ 
    Takes a square matrix and calculates it's inverse through LU Decomposition
    Parameters: matrix "A" to do LU Decomposition on

    Returns: Matrix "A_inv" which is the inverse matrix to the parameter "A"
    """

    #Ax=I, x=A^-1, LUx=I, Ld=I, Ux=d

    #Determine if the matrix is a square and if the A is a singular matrix
    if int(np.linalg.det(A)) == 0:
        return error

    nrow,ncol = A.shape
    if nrow != ncol:
        return error
    else:
        #To find A's inverse use the identity matrix as the solution to your system of equations, thus solving for x yields the identity matrix
        I = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])

        L, U = naive_LU(A)

        #Solve for inverse using systems of equations 
        A_inv = solve_LU(L, U, I)
    
    return A_inv

#Problem 12 - Richardson method for solving x
def  Richardson_it(A, b, omega, tol, max_it):

    """
    Utilizes the Richardson's interative method to find a solution x
    Parameters: Matrix "A", matrix "b" the solution, a constant "omega", tolerance "tol", and maximum iterations "max_it"
    
    Returns: The function returns the final solution "x", the number of iterations used "n_it", and the final error "err"
    """
    
    # Check if the convergence condition is satisfied
    if np.linalg.norm(np.eye(len(A)) - (omega * A), ord=2) >= 1:
        return error

    n_it = 0
    x = np.zeros(b.shape)
    #We don't know initial error so for make it large for the loop to work
    err = np.inf
    while err >= tol and n_it < max_it:
        x_new = x + omega * (b - np.dot(A,x))
        err = np.linalg.norm(np.dot(A,x) - b, ord=2)
        x = x_new
        n_it += 1

    return x, n_it, err

#Problem 15 - Finding the largest eigenvalue
def largest_eig(A, tol, max_it):

    """
    Utilizes the power method to find the maximum eigenvalue of a system
    Parameters: Matrix "A" being applied to x, the tolerance "tol", and maximum iterations "max_it"
    
    Returns: The function returns the largest Eigen value "eig", the final x used "x", the number of iterations used "n_it", and the final error "err"
    """

    #We don't know initial error so make it large for the loop to work
    err = np.inf
    #Initial test x
    x = np.ones((len(A),1)) 
    x_max = x
    eig_old = 1
    eig_max = 1
    n_it = 0

    while n_it < max_it and err >= tol:
        x_new = np.dot(A,x)
        #Largest lambda in x
        eig = np.max(x_new)

        err = np.abs((eig - eig_old)/eig)
        eig_old = eig
        n_it += 1

        #Update largest lambda and eigen vector
        if eig > eig_max:
            eig_max = eig
            x_max = x_new
        #Normalize x for next iteration
        x = x_new / eig

    #Set the return to the max eigenvalue
    eig = eig_max
    x = x_max
    return eig, x, n_it, err

##########################################################################
##########################################################################
#
# BONUS
#
##########################################################################
##########################################################################

def my_Cholesky(A):

    """
    Utilizes the Cholesky method to perform a special LU decomposition of a symmetrical matrix
    Parameters: Matrix "A" which will be operated upon by Cholesky factorization
    
    Returns: The function returns the U part of the final decomposition
    """

    #Check symmetry first 
    if not np.allclose(A, np.transpose(A)):
        return error
    
    U = np.zeros(A.shape)

    for i in range(len(A)):
        for j in range(i+1):
            #If at i,i follow what is below else operate using i,j operations
            if i == j:
                U[i, i] = np.sqrt(A[i, i] - np.sum(U[i, :i]**2))
            else:
                U[i, j] = (A[i, j] - np.sum(U[i, :j] * U[j, :j])) / U[j,j]

    #Transpose for the correct factor "Upper"
    U = U.T
    return U

def my_Gauss_Siedel(A, b, tol, max_it):

    """
    Utilizes the Gauss Siedel method to perform a an iterative method for solving for x
    Parameters: Matrix "A" which will be operated upon, "b" the solution to the system, "tol" the tolerance, and "max_it" the maximum iteration number
    
    Returns: The array "x" which is the solution to the system
    """

    # Setting the initial guess for x
    x = np.zeros(len(b)) 
     
    #Keeping track of iterations
    for j in range(max_it):
        x_new = np.copy(x)
        for i in range(len(b)):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        
        #Checking for error and its state within tolerance
        if np.linalg.norm(np.dot(A, x_new) - b, ord=2) < tol:
            return x_new
        
        x = x_new

    return x


    

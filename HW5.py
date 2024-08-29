#File - HW5.py
#Author - Jacky Lin
#UT EID - jyl866
#Course - COE311K

import numpy as np
import matplotlib.pyplot as plt
error = "error"

# Problem 2
def my_finite_diff(fx, xi, fd_type):

    """
    Performs a finite difference approximation based on the user choice of a forward, backward, or centered finite difference method
    Parameters: "fx" numpy vector of values of a function f, "xi" numpy vector of the location of where we have fx values at, and "fd_type" a string containing the type of finite difference approximation the user wants

    Returns: "dfxi" a numpy vector containing the finite difference approximations of each interior points of xi 
    """
    # Check if the lengths of fx and xi are the same
    if len(fx) != len(xi):
        return error
    
    #Creating the ouput vector of interior points which is length of xi and fx - 2
    dfxi = np.zeros(len(xi) - 2)

    #Updating the vector using the backward finite difference equation
    if fd_type == "Backward":
        for x in range(1, len(xi) - 1):
            dfxi[x - 1] = (fx[x] - fx[x - 1]) / (xi[x] - xi[x - 1])
    elif fd_type == "Forward":
    #Updating the vector using the forward finite difference equation

        for x in range(1, len(xi) -1):
            dfxi[x - 1] = (fx[x + 1] - fx[x]) / (xi[x + 1] - xi[x])
    elif fd_type == "Centered":
    #Updating the vector using the centered finite difference equation

        for x in range(1, len(xi) - 1):
            dfxi[x - 1] = (fx[x + 1] - fx[x - 1]) / (xi[x + 1] - xi[ x - 1])
    else: 
        return "error"
    
    return dfxi

# Problem 3 
def fourth_order_diff(f, xi, h):

    """
    Calculates the first derivative at a given set of points using a prescribed height according to a fourth order accurate formula
    Parameters: "f" the function we find the first derivative of, "xi" locations we wish to approximate at, "h" the distance each point is seperated by

    Returns: "dfxi" a numpy vector containing the first derivative approximations of each point of xi 
    """

    # h must be greater than 0
    if h <= 0:
        return error
    
    n = len(xi) 
    dfxi = np.zeros(n)

    # Computing the first derivative based on the formula
    for i in range(len(xi)):
        dfxi[i] = (-f(xi[i] + 2*h) + 8*f(xi[i] + h) - 8*f(xi[i] - h) + f(xi[i] - 2*h)) / (12 * h)
    
    return dfxi

# Problem 5
def my_composite_trap(x, fx):

    """
    Calculates the integral using composite trapezoid method
    Parameters: "x" the vector of inputs to the function, "fx" the vector of outputs of the function

    Returns: "I" a value approximating the area of the function using composite trapezoid function
    """

    # Check if inputs have the same length
    if len(x) != len(fx):
        return error
    
    #Set initial I to zero and then add based on spaces
    I = 0

    #Use trapazoid approximation for integral on each point
    for i in range(len(x) - 1):
        I += 0.5 * (fx[i] + fx[i + 1]) * (x[i + 1] - x[i])
    
    return I

# Problem 7
def solve_freefall_RK4(x0, v0, nt, dt, g, cd, m):

    """
    Calculates the velocity and position at a point of time by using a RK4 scheme
    Parameters: "x0" : Initial position, "v0": Initial velocity, "nt": Number of time steps, "dt": Time step size, "g": Gravitational constant, "cd": Coefficient of drag, and "m": Mass of object
    
    Returns: "x" a vector of all positions at each successive timestep, and "v" a vector of all the velocities at each successive timestep
    """

    #Creating the output vectors for each time step
    x = np.zeros(nt + 1)
    v = np.zeros(nt + 1)

    #Setting initial values to the output
    x[0] = x0
    v[0] = v0
    
    for i in range(1, nt + 1):
        #Solve for each k coefficient using partial derivatives given in the problem using values previoud to your calculation
        #Since x is a function of v, each k for x uses the previous k for v
        k1x = v[i-1]
        k1v = g - (cd / m) * (v[i-1])**2
        
        k2x = v[i-1] + 0.5 * dt * k1v
        k2v = g - (cd / m) * (v[i-1] + 0.5 * dt * k1v)**2
        
        k3x = v[i-1] + 0.5 * dt * k2v
        k3v = g - (cd / m) * (v[i-1] + 0.5 * dt * k2v)**2
        
        k4x = v[i-1] + dt * k3v
        k4v = g - (cd / m) * (v[i-1] + dt * k3v)**2
        
        #Apply the fourth order formula for Runge-Kutta and update the output vector for both x and v
        x[i] = x[i-1] + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        v[i] = v[i-1] + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

    return x, v

# Problem 8
def solve_BVP_FD(T0, T1, k, dx):
    """
    Solves a boundary value differential equation temperature problem using centered finite differences
    Parameters: "T0": Temperature at left boundary, "T1": Temperature at right boundary, "k": The coefficient that shows up in the equation, "dx": The desired spacing for the finite difference approximation
    
    Returns: "T" the estimated solution at each successive timestep
    """

    #Finding number of points
    N = int(1 / dx) + 1
    
    # Creating a b vector according to the centered finite difference equation
    b = np.zeros(N)
    for i in range(1, N):
        b[i] = dx**2 *((-i*dx)/k)
    b[N - 1] = 1

    # Creating the A coefficient matrix using the centered finite difference equation 
    A = np.zeros((N, N))

    for i in range(1, N - 1):
        A[i, i] = -2 
        A[i, i - 1] = 1
        A[i, i + 1] = 1
    
    A[0, 0] = 1
    A[N - 1, N - 1] = 1

    # Solve the system of equations for T
    T = np.linalg.solve(A, b)
    
    return T


#Setting the constants
x0 = 0
v0 = 0
nt = 5
g = 9.81
cd = 0.25
m = 68.1

#Creating the time step, position, and velocity vectors
dt = [2, 1, 0.5, 0.25, 0.125, 0.0625]
x_Calculated = np.zeros((len(dt), len(dt)))
v_Calculated = np.zeros((len(dt),len(dt)))
t = np.linspace(0, 10, nt+1)

v_real = np.sqrt(g*m/cd)*np.tanh(np.sqrt(g*cd/m)*t)
x_real = m/cd*np.log(np.cosh(np.sqrt(g*cd/m)*t))

#Creating the error vectors
errorx = np.zeros(len(dt))
errorv = np.zeros(len(dt))

#Updating the error
for i in range(0, len(dt)):
    x, v = solve_freefall_RK4(x0, v0, nt, dt[i], g, cd, m)
    x_Calculated[i][:] = x
    v_Calculated[i][:] = v

    exactx = x_real[len(x_real) - 1]
    exactv = v_real[len(v_real) - 1]
    errorx[i] = np.abs(x_Calculated[i][len(dt)-1] - exactx)
    errorv[i] = np.abs(v_Calculated[i][len(dt)-1] - exactv)
    
plt.plot(dt, errorx, marker = "o")
plt.title("Position: dt vs Error")
plt.xlabel("dt")
plt.ylabel("Error")
plt.show()
plt.plot(dt, errorv, marker = "o")
plt.title("Velocity: dt vs Error")
plt.xlabel("dt")
plt.ylabel("Error")
plt.show()
#File - HW3.py
#Author - Jacky Lin
#UT EID - jyl866
#Course - COE311K

import numpy as np
error = "error"
import matplotlib.pyplot as plt

#Problem 2 
def least_squares_poly(x, y, k):

    """ 
    Performs a least squares regression on a general polynomial
    Parameters: "x" the data inputs, "y" the outputs you want to fit to, and "k" the order of the polynomial

    Returns: "a" a vector containing the coefficients that defines the polynomial and minimize error
    """
    #Z must be length of amount of data but columns equal to order + 1
    Z = np.zeros((len(x), k + 1))

    #Updating Z with f(x) values from the polynomial 
    for i in range(k + 1):
        Z[:, i] = x ** i
    
    #Operating for finding coefficients through ZTZa = ZTY
    ZT = Z.T
    ZTZ = np.dot(ZT, Z)
    ZTY = np.dot(ZT, y)
    a = np.linalg.solve(ZTZ, ZTY) 
    
    return a

#Problem 3
def least_squares_fourier(x, y, k, omega_o):

    """ 
    Performs a least squares fourier regression on a general fourier series
    Parameters: "x" the data inputs, "y" the outputs you want to fit to, and "k" the number of frequencies, and "omega_o" the fundamental frequency"

    Returns: "a" a vector containing the coefficients that defines the fourier series and minimize error
    """

    n = len(x)
    #Construct the matrix Z with 2k + 1
    Z = np.zeros((n, (2 * k) + 1))

    #Cos and Sin alternate in Z for all the rows if k > 0, while first column are zeros
    for i in range(n):
        Z[i][0] = 1
        for j in range(1, k + 1):
            Z[i][(2 * j) - 1] = np.cos(j * omega_o * x[i])
            Z[i][2 * j] = np.sin(j * omega_o * x[i])

    #Computing the normal equations and solving for a using ZTZa = ZTY
    ZT = Z.T
    ZTZ = np.dot(ZT, Z)
    ZTY = np.dot(ZT, y)
    a = np.linalg.solve(ZTZ, ZTY)
    
    return a

#Problem 4
def my_dft(x):

    """ 
    Performs a discrete squares fourier transform on a set of data "x" 
    Parameters: "x" the vector of data that are the outputs you want to perform Fourier analysis on

    Returns: "F" a vector containing the Fourier coefficients 
    """
    n = len(x)
    #Creating a F array of length n for all coefficients containing Complex Numbers
    F = np.zeros(n, dtype = complex)
    
    for k in range(n):
        for l in range(n):
            #The definition we are using requires a sum of the data so we use +=
            F[k] += x[l] * np.exp(-(1j)* k * ((2*  np.pi)/ n) * l)

    return F

#problem 6
def my_poly_interp(x, fx, xi, ptype):

    """ 
    Performs interpolation using Newton or Lagrange polynomials based on user input
    Parameters: "x" the input data, "fx" the data you want to interpolate, "xi" the coordinates you want to interpolate at, and "ptype" the user input

    Returns: "fxi" the interpolated values at each point xi
    """
    n = len(x)
    m = len(xi)
    fxi = np.zeros(m)

    if ptype == "Newton":
        #We make a Divided Differences table as seen in the textbook where the first row are our coefficients 
        n = len(fx)
        b = np.zeros([n, n])
        #First column are our y's
        b[:,0] = fx

        #Creating all of the difference elements in the table starting in second column
        for i in range(1,n):
            for j in range(n - i):
                b[j][i] = (b[j+1, i-1] - b[j, i-1]) / (x[j+i] - x[j])

        #Adding first coefficient 
        fxi += b[0, 0]

        #Applying the coefficient to the polynomials for each xi
        z = np.ones(m)
        for k in range(1, n):
            #up to x(n-1)
            z *= (xi - x[k - 1])
            fxi += b[0, k] * z
    
        return fxi
    
    if ptype == "Lagrange":
        #Loop for each data point xi
        for k in range(m):
            #Creating the polynomial of order n with all L's
            L = np.ones(n)
            for j in range(n):
                for i in range(n):
                    #xj can't be equal to xi, else create L
                    if j != i:
                        L[j] *= (xi[k] - x[i]) / (x[j] - x[i])
            #Update output with interpolated result 
            fxi[k] = np.sum(L * fx)
        return fxi
    
#Problem 5
n = 64
fs = 128
#Creating a function to output data based on the given function
def f(t):
    return 1.5 + (1.8 * np.cos(2 * np.pi * 12 * t)) + (0.8 * np.sin(2 * np.pi * 20 * t)) - (1.25 * np.cos(2 * np.pi * 28 * t))

#deltaT = 1/fs
#deltaT = T/n
delta_t = (1/fs)
time_period = n/fs

delta_f = fs/n
f_min = 1/(time_period)
f_max =  (0.5) * (fs)

print("Time period between each data point is:", delta_t)
print("Time period is:", time_period)
print("Spacing between frequency is:", delta_f)
print("Minumum frequency is:", f_min)
print("Maximum frequency is:", f_max)

#Plotting Signal vs Time
t = np.arange(0, time_period, delta_t)
data = f(t)

plt.plot(t, data)
plt.title("Signal vs Time")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()

#Plotting Amplitude vs Frequency 
dft = my_dft(data)
dft = np.abs(dft)
Frequency = np.arange(n)/time_period
plt.bar(Frequency, dft)

plt.title("Amplitude vs Frequency")
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.show()


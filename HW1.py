import numpy as np
import matplotlib.pyplot as plt

#problem 1
def exact_velocity(c_d, m, t, g):
    v = []
   
    for x in range(len(t)):
        v.append((np.sqrt((g*m)/c_d)) * (np.tanh(np.sqrt(((g*c_d)/m))*t[x])))

    return v 

#problem 3
def  forward_Euler_velocity(c_d, m, t, g):
    x = len(t)
    v = [0]
    i = 1

    while i < x:
        dt = t[i]-t[i-1]
        v.append((v[i-1]) + (dt * (g-((c_d/m)*(v[i-1])**2))))
        i+=1
        
    return v

#problem 5
#Code in PDF as it is not a function
import numpy as np 
import matplotlib.pyplot as plt
from Frame import nodes

xFac = 100


xCoord = np.array([])
yCoord = np.array([])

# Funksjon som henter ut n index i liste. 
def Extraxt(lst,n):
    return [item[n] for item in lst]

x = Extraxt(nodes,0)
y = Extraxt(nodes,1)

Fy = Extraxt(S_local,1)

print(Fy)

plt.plot(x,y)
plt.ylabel('y-coord')
plt.xlabel('x-coord')
plt.plot(x,Fy)

"""
fig, ax = plt.subplots()  # Create a figure containing a single axes.

ax.plot(x,y)  # Plot some data on the axes.


fig = plt.figure()

axes = fig.add_axes([0,0,1,1])
fig.gca().set_aspect('equal', adjustable ='box')
"""

import numpy as np 
import math
from matplotlib import pyplot as plt
import pandas

add = np.array([[1,  1, 1],
                [1,  1, 1],
                [1,  1, 1]])

dda = np.array([[2,  2, 2],
                [2,  2, 2],
                [2,  2, 2]])

Mat = np.zeros([6,6])
Mat2 = Mat[0:3,3:6] = Mat[0:3,3:6] + add
Mat3 = Mat[3:6,0:3] = Mat[3:6,0:3] + dda



# Structure
L = 10
H = 5 

# Left wall
leftWall = np.array([[0,0],[0,H]])
topPlate = np.array([[0,H],[L,H]])
rigthWall = np.array([[L,H],[L,0]])


L = 10
mesh = 1

pts = np.array([])
# Function that splits the main structure into smaller elements (coordiantes)
x = np.arange(0,L,mesh)


# Function that creates the elements between each node

ptLen = len(pts)

ptIndex = np.arange(0,ptLen,1)

p1 = np.array([])
p2 = np.array([])


for i in range(len(pts)):
    p1= np.append(p2,i)
    p2 = np.append(p2,i+1)
    
    
    
    
NodeX = np.array([])

for i in x:
    NodeX = np.append(NodeX,i)
    NodeX = np.append(NodeX,0, axis = 1)

nodes = np.array([[0,0],
                [5,0],
                [10,0]])
    
    
import numpy as np 
import math
from matplotlib import pyplot as plt
import copy
import pandas


# INPUT DATA
# -------------------
# Constants
E = 1#200*10**9  #[N/m2]
A = 1 #0.005 # [m2]
I = 1 # [mm4]
xFac = 100 # Scale factor for plotted displacement

# Nodal coordinates [x,y], in ascending node order
nodes = np.array([[0,0],
                [5,0],
                [10,0]])

# Members [node_i, node_j]
members = np.array([[1,2],
                   [2,3]])

# Supports
restrainedDoF = [1,2,3,8,9] # The degrees of freedom restrained by supports


# Loading
#forceVector = np.array([[0,-200000,0,0,0,0,0,0,0,0,0,0]]).T # Vector of externally applied loads
forceVector = np.array([np.zeros(9)]).T
#forceVector[3] = -500000
forceVector[4] = -10
#forceVector[7] = -100000
# END OF DATA ENTRY
# -------------------

# Calculate member orientation and length
#----------------------------------------
# Define a function to calculate the member orientation and length
def memberOrientation(memberNo):
    memberIndex = memberNo-1 #Index identifying member in array of members
    node_i = members[memberIndex][0] # First node of the member
    node_j = members[memberIndex][1] # Second node of the member

    xi = nodes[node_i-1][0] # X-coord of node i
    yi = nodes[node_i-1][1] # Y-coord of node i

    xj = nodes[node_j-1][0] # X-coord of node j
    yj = nodes[node_j-1][1] # Y-coord of node j

    # Angle of member with respect to horizontal axis

    dx = xj-xi #x-component of member
    dy = yj-yi #y-component of member

    mag = math.sqrt(dx**2+dy**2) # Magnitude of vector (length of member)
    memberVector = np.array([dx,dy]) # Member represented as a vector

    # Need to capture quadrant first, then identify appropriate reference axis and offet angle
    if (dx>0 and dy==0):                                    #X-axis pos
        theta = 0
    elif (dx==0 and 0<dy):                                  #Y-axis pos
        theta = math.pi/2
    elif (dx<0 and dy==0):                                  #X-axis neg
        theta = math.pi
    elif (dx==0 and dy<0):                                  #Y-axis neg
        theta = 3*math.pi/2

    elif (0<dx and 0<dy):            
        #0<theta<90                   
        refVector = np.array([1,0])                         #Vector describing the positive x-axis
        theta = math.acos(refVector.dot(memberVector)/(mag))  #Standard formula for angle between 2 vectors

    elif (dx<0 and dy>0):  
        #90<theta<180                                 
        refVector = np.array([0,1])                         #Vector describing the positive y-axis
        theta = math.acos(refVector.dot(memberVector)/(mag)) + (math.pi/2)

    elif (dx<0 and dy<0):         
        #180<theta<270                         
        refVector = np.array([-1,0])                         #Vector describing the positive y-axis
        theta = math.acos(refVector.dot(memberVector)/(mag)) + (math.pi)

    elif (0<dx and dy<0):            
        #270<theta<360                       
        refVector = np.array([0,-1])                         #Vector describing the positive y-axis
        theta = math.acos(refVector.dot(memberVector)/(mag)) + (3*math.pi/2)

    return [theta, mag]

# Calculate orientation and length for each member and store
orientations = np.array([]) # Initialise an array to hold orientations
lengths  = np.array([]) # Initialise an array to hold member lengths

for n, mbr in enumerate(members):
    [angle, length] = memberOrientation(n+1) # Member 1, not index 0, because first n = 0
    orientations = np.append(orientations,angle)
    lengths = np.append(lengths,length)


# Define a function to calculate member global stiffness matrix
#--------------------------------------------------------------
def calculateKg(memberNo):
    theta = orientations[memberNo-1]
    L = lengths[memberNo-1]

    c = math.cos(theta)
    s = math.sin(theta) 

    K11 = E*I/L**3 * np.array([  [(A*L**2/I)*c**2 + 12*s**2,    (A*L**2/I-12)*c*s,         -6*L*s], 
                                 [(A*L**2/I - 12)*c*s,       A*L**2/I*s**2 + 12*c**2,   6*L*c ],
                                 [-6*L*s,                   6*L*c,                      4*L**2]   ])

    K12 = E*I/L**3 * np.array([  [-(A*L**2/I*c**2 + 12*s**2),    -(A*L**2/I - 12)*c*s,          -6*L*s], 
                                 [-(A*L**2/I - 12)*c*s,          -(A*L**2/I*s**2 + 12*c**2),     6*L*c],
                                 [6*L*s,                          -6*L*c,                        2*L**2]   ])
    
    K21 = E*I/L**3 * np.array([  [-(A*L**2/I*c**2 + 12*s**2),    -(A*L**2/I - 12)*c*s,           6*L*s], 
                                 [-(A*L**2/I - 12)*c*s,          -(A*L**2/I*s**2 + 12*c**2),    -6*L*c],
                                 [-6*L*s,                          6*L*c,                        2*L**2]   ])

    K22 = E*I/L**3 * np.array([  [A*L**2/I*c**2 + 12*s**2,    (A*L**2/I-12)*c*s,          6*L*s], 
                                 [(A*L**2/I - 12)*c*s,       A*L**2/I*s**2 + 12*c**2,  -6*L*c ],
                                 [6*L*s,                    -6*L*c,                     4*L**2]   ])


    return (K11, K12, K21, K22)

# Define a function to calculate member local stiffness matrix
#--------------------------------------------------------------
def calculate_K_local(memberNo):
    L = lengths[memberNo-1]


    K11 = E*I/L**3 * np.array([  [A*L**2/I,    0,          0], 
                                 [0,         12,        6*L],
                                 [0,         6*L,     4*L**2]   ])

    K12 = E*I/L**3 * np.array([  [-A*L**2/I,    0,         0], 
                                 [0,          -12,       6*L],
                                 [0,          -6*L,   2*L**2]   ])
    
    K21 = E*I/L**3 * np.array([  [-A*L**2/I,    0,         0], 
                                 [0,          -12,      -6*L],
                                 [0,          6*L,    2*L**2]   ])

    K22 = E*I/L**3 * np.array([  [A*L**2/I,    0,         0], 
                                 [0,          12,     -6*L ],
                                 [0,        -6*L,    4*L**2]   ])

    k = np.zeros([6,6])
    k[0:3,0:3] = k[0:3,0:3] + K11
    k[0:3,3:6] = k[0:3,3:6] + K12
    k[3:6,0:3] = k[3:6,0:3] + K21
    k[3:6,3:6] = k[3:6,3:6] + K22

    return (k)

# Build primart stiffness matrix, Kp
#-----------------------------------
nDoF = np.amax(members)*3 # Total numbers of degrees of freedom in the problem
Kp = np.zeros([nDoF,nDoF]) # Initialising the primary stiffness matrix

for m, mbr in enumerate(members):
    [K11, K12, K21, K22] = calculateKg(m+1)

    node_i = mbr[0] # Node number for node i of this member
    node_j = mbr[1] # Node number for node j of this member

    ia = 3*node_i-3     # Index starts at 0 (ex. v6 = index 5)
    ib = 3*node_i-2
    ic = 3*node_i-1
    ja = 3*node_j-3
    jb = 3*node_j-2
    jc = 3*node_j-1

    Kp[ia:ic+1,ia:ic+1] = Kp[ia:ic+1,ia:ic+1] + K11  # Ex Kp[y = 0-3, x 0-3], dvs. [rows, columns]
    Kp[ia:ic+1,ja:jc+1] = Kp[ia:ic+1,ja:jc+1] + K12  # Ex Kp[y = 0-3, x 3-6]
    Kp[ja:jc+1,ia:ic+1] = Kp[ja:jc+1,ia:ic+1] + K21
    Kp[ja:jc+1,ja:jc+1] = Kp[ja:jc+1,ja:jc+1] + K22


# Extract the structure stiffness matrix, Ks
#---------------------------------------------
restrainedIndex = [x-1 for x in restrainedDoF]    

# Reduce the structure stiffness matrix by deleting the rows and columns for restrained DoF
Ks = np.delete(Kp, restrainedIndex, 0) # Delete rows
Ks = np.delete(Ks, restrainedIndex, 1) # Delete columns
Ks = np.matrix(Ks) # Convert from nump array to a matrix


# Solve unknow displacements
# --------------------------------------------
forceVectorRed = copy.copy(forceVector)
forceVectorRed = np.delete(forceVectorRed, restrainedIndex,0)

U = np.linalg.inv(Ks)*forceVectorRed

# Solve for reactions 
# ---------------------------------------------
UG = np.zeros(nDoF)


c = 0
for i in np.arange(nDoF):
    if i in restrainedIndex:
        UG[i] = 0
    else:
        UG[i] = U[c]
        c = c + 1
        
UG = np.array([UG]).T
FG = np.matmul(Kp,UG)


# Solve for member forces
# --------------------------------------------
mbrForces = np.array([]) # Initialize an array to hold member forces

for n, mbr in enumerate(members):
    theta = orientations[n]
    mag = lengths[n]
    
    #node_i = members[n][0] # For kontroll av enkeltelement
    #node_j = members[n][1]

    node_i = mbr[0]
    node_j = mbr[1]
    
    ia = 3*node_i-3     # Index starts at 0 (ex. v6 = index 5)
    ib = 3*node_i-2
    ic = 3*node_i-1
    ja = 3*node_j-3
    jb = 3*node_j-2
    jc = 3*node_j-1

    # Transformation matrix
    c = math.cos(theta)
    s = math.sin(theta)
    t_r = np.array([[c,  s, 0],
                    [-s, c, 0],
                    [0,  0, 1]])
  
    T = np.zeros([6,6])
    T[0:3,0:3] = T[0:3,0:3] + t_r
    T[3:6,3:6] = T[3:6,3:6] + t_r
    
    k_local = calculate_K_local(n+1)    

    # Retrieve global displacement for element
    disp = np.array([[ UG[ia], UG[ib], UG[ic], UG[ja], UG[jb], UG[jc] ]])
    # Transform global displacement to local
    disp_local = np.matmul(T, disp)[0]  
    # Calculate local element forces
    S = np.matmul(k_local,disp_local)
    
    mbrForces = np.append(mbrForces,S)
    
    mbrForces = np.round(mbrForces,2)


        
# Calculate how many arrayes needed to split mbrForces into arrays of 3, ie. forces in each element node. 
forceLen = mbrForces.size/3


# Split member forces into arrays of forces in each node. 
S_local = np.split(mbrForces, forceLen, axis=0)

    
    
print(S_local)
print('test')
"""
x = np.array(['Fx1', 'Fy1', 'M1', 'Fx2', 'Fy2', 'M2'])
y = 'test'

Sl = pandas.DataFrame(S, columns = [''], index=['Fx1', 'Fy1', 'M1', 'Fx2', 'Fy2', 'M2'])











print(FG)




x = np.arange(1,nDoF+1,1)
y = np.arange(1,nDoF+1,1)

Kp = np.round(Kp,2)
df = pandas.DataFrame(Kp, columns=x, index=y)

print(df)
"""
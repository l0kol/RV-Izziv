import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from funkcije import *
from vrednosti import *



#Kamera 1:
rot_kamera1 = rot(35, 0, -45)
trans_kamera1 = transl(510, 240, 370)
K1 = np.array([[376.67782593,   0,   323.34994012],[  0, 373.62069702, 229.49620298],[  0, 0,1 ]])
P1 = np.hstack((rot_kamera1, trans_kamera1))
P1 = np.delete(P1, 2, 1)
H1 = np.dot(K1, P1)
distR = np.array([[-0.39244406,  0.27547218, -0.0042847,   0.01512155, -0.1079625 ]])


#Kamera 2:
rot_kamera2 = rot(40, 0, 46)
trans_kamera2 = transl(80, 240, 370)
K2 = np.array([[113.70654297,   0,  299.20507805], [0, 127.08498383, 356.26959421], [  0,           0,       1]])
K2_ = np.array([[263.32574463,   0,   375.4140521 ],[  0,  248.37150574, 214.63840549], [  0, 0, 1]])
P2 = np.hstack((rot_kamera2, trans_kamera2))
P2 = np.delete(P2, 2, 1)
H2 = np.dot(K2, P2)
distL = np.array([-0.47394711,  0.42445038, -0.01454484, -0.00168144, -0.08305112])


# #Kamera 3:
rot_kamera3 = rot(0, 0, 0)
trans_kamera3 = transl(295, 150, 495)
K3 = np.array([[379.57342529, 0, 331.19525784],[  0,  376.40127563, 253.93459374], [  0,   0,  1,  ]])
P3 = np.hstack((rot_kamera3, trans_kamera3))
P3 = np.delete(P3, 2, 1)
H3 = np.dot(K3, P3)
distC = np.array([[-0.41526578,  0.29828027, -0.0038376,   0.00100217, -0.20197799]])



tocke_3D_1 = []
tocke_3D_2 = []
tocke_3D_3 = []
tocke_3D = []


P1 = np.hstack((rot_kamera1, trans_kamera1))
P2 = np.hstack((rot_kamera2, trans_kamera2))
P3 = np.hstack((rot_kamera3, trans_kamera3))



######################### Popravi popačenje/distorzijo: #########################
enke = np.ones(len(camDesnaX), dtype = int)
desnaXY = np.array(list(zip(camDesnaX, camDesnaY)))
desnaXY_u = cv.undistortPoints(np.array(desnaXY, dtype=float), K1, distCoeffs=distR) 

levaXY = np.array(list(zip(camLevaX, camLevaY)))


levaXY_u = cv.undistortPoints(np.array(levaXY, dtype=float),
 K2_, distCoeffs=distL) 

centerXY = np.array(list(zip(camCentX, camCentY)))
centerXY_u = cv.undistortPoints(np.array(centerXY, dtype=float), K3, distCoeffs=distC) 


################# Izračunaj 3D trajektorijo ################################
for i in range(len(camDesnaX)):
    point1 = np.array([desnaXY_u[i][0][0], desnaXY_u[i][0][1], 1])
    proj1 = np.dot(H1, point1)
    projnorm1 = proj1 * 1

    point2 = np.array([levaXY_u[i][0][0], levaXY_u[i][0][1], 1])
    proj2 = np.dot(H2, point2)
    projnorm2 = proj2 * 1

    point3 = np.array([centerXY_u[i][0][0], centerXY_u[i][0][1], 1])
    proj3 = np.dot(H3, point3)
    projnorm3 = proj3 * 1

    p = triangulate_nviews([P1, P2, P3], [np.array(projnorm1), np.array(projnorm2), np.array(projnorm3)])
    # print('Projected point from 3 camera views:',  p)
    tocke_3D.append(p)

    tocke_3D_1.append(projnorm1)
    tocke_3D_2.append(projnorm2)
    tocke_3D_3.append(projnorm3)



x = [list[1] for list in tocke_3D_1]
y = [list[0] for list in tocke_3D_1]
z = [list[2] for list in tocke_3D_1]

x2 = [list[1] for list in tocke_3D_2]
y2 = [list[0] for list in tocke_3D_2]
z2 = [list[2] for list in tocke_3D_2]

x3 = [list[1] for list in tocke_3D_3]
y3 = [list[0] for list in tocke_3D_3]
z3 = [list[2] for list in tocke_3D_3]

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# # defining axes

# ax.scatter(x, y, z, c='red')
# ax.scatter(x2, y2, z2, c='blue')
# ax.scatter(x3, y3, z3, c='green')

# ax.scatter(trans_kamera1[0], trans_kamera1[1], trans_kamera1[2], c='red', s=100)

# ax.scatter(trans_kamera2[0], trans_kamera2[1], trans_kamera2[2], c='blue', s=100)

# ax.scatter(trans_kamera3[0], trans_kamera3[1], trans_kamera3[2], c='green', s=100)
# #ax.view_init(-170, 120)
# #ax.plot3D(x, y, z, 'green')
# # syntax for plotting
# ax.set_title('3D trajektorija')
# ax.axis('off')
# plt.show()


x = [list[0] for list in tocke_3D]
y = [list[1] for list in tocke_3D]
z = [list[2] for list in tocke_3D]

fig = plt.figure()
# syntax for 3-D projection
ax = plt.axes(projection='3d')

# defining axes

ax.scatter(x, y, z, c="purple")
#ax.view_init(-170, 120)
#ax.plot3D(x, y, z, 'green')
# syntax for plotting
ax.set_title('3D trajektorija')
ax.axis('off')
plt.show()
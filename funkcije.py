import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npla

# Nalozi sliko
def loadImage(iPath):
    oImage = np.array(im.open(iPath))
    return oImage

# Prikazi sliko
def showImage(iImage, iTitle=''):
    plt.figure()
    plt.imshow(iImage, cmap = 'gray')
    plt.suptitle(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')
    
# Pretvori v sivinsko sliko
def colorToGray(iImage):
    dtype = iImage.dtype
    r = iImage[:,:,0].astype('float')
    g = iImage[:,:,1].astype('float')
    b = iImage[:,:,2].astype('float')
    
    return (r*0.299 + g*0.587 + b*0.114).astype(dtype)


def rot(alpha, beta, gama):

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gama)
    sg = np.sin(gama)
    rot_x = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    rot_y = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rot_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    return np.dot(rot_x, np.dot(rot_y, rot_z))


def transl(dx, dy, dz):
    # return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])
    return np.array([[dx], [dy],[dz]])


#https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
def triangulate_nviews(P, ip):
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]

def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)



def midpoint_triangulate(x, cam):
    """
    Args:
        x:   Set of 2D points in homogeneous coords, (3 x n) matrix
        cam: Collection of n objects, each containing member variables
                 cam.P - 3x4 camera matrix
                 cam.R - 3x3 rotation matrix
                 cam.T - 3x1 translation matrix
    Returns:
        midpoint: 3D point in homogeneous coords, (4 x 1) matrix
    """

    n = len(cam)                                         # No. of cameras

    I = np.eye(3)                                        # 3x3 identity matrix
    A = np.zeros((3,n))
    B = np.zeros((3,n))
    sigma2 = np.zeros((3,1))

    for i in range(n):
        a = -np.transpose(cam[i].R).dot(cam[i].T)        # ith camera position
        A[:,i,None] = a

        b = npla.pinv(cam[i].P).dot(x[:,i])              # Directional vector
        b = b / b[3]
        b = b[:3,None] - a
        b = b / npla.norm(b)
        B[:,i,None] = b

        sigma2 = sigma2 + b.dot(b.T.dot(a))

    C = (n * I) - B.dot(B.T)
    Cinv = npla.inv(C)
    sigma1 = np.sum(A, axis=1)[:,None]
    m1 = I + B.dot(np.transpose(B).dot(Cinv))
    m2 = Cinv.dot(sigma2)

    midpoint = (1/n) * m1.dot(sigma1) - m2        
    return np.vstack((midpoint, 1))



import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import PIL.Image as im
from funkcije import *
import os


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,7)
frameSize = (640,480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.
imgpointsC = [] # 2d points in image plane.


#imagesLeft = sorted(glob.glob('Desktop\\test\\Kamera1\\*.png'))
imagesLeft = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/leva/*.png")))
imagesRight = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/desna/*.png")))
imagesCenter = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/center/*.png")))

for imgLeft, imgRight, imgCenter in zip(imagesLeft, imagesRight, imagesCenter):

    imgL = cv.imread(imgLeft)
    print("Left")
    print(imgL.shape)
    imgR = cv.imread(imgRight)
    print("Right")
    print(imgR.shape)
    imgC = cv.imread(imgCenter)
    print("Center")
    print(imgC.shape)

    grayL = colorToGray(imgL)
    grayR = colorToGray(imgR)
    grayC = colorToGray(imgC)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    retC, cornersC = cv.findChessboardCorners(grayC, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if not retC:
        cv.drawChessboardCorners(imgC, chessboardSize, cornersC, retC)

    if retC == True:

        diff = cornersC[chessboardSize[0]-1][0][0] - cornersC[0][0][0]
        if diff > 0:
            cornersC = cornersC[::-1].astype(np.float32)

        objpoints.append(objp)

        cornersC = cv.cornerSubPix(grayC, cornersC, (11,11), (-1,-1), criteria)
        imgpointsC.append(cornersC)

        # Draw and display the corners
        cv.drawChessboardCorners(imgC, chessboardSize, cornersC, retC)
        cv.imshow('img center', imgC)
        cv.waitKey(1000)


cv.destroyAllWindows()


############## kalibracija #######################################################

retC, cameraMatrixC, distC, rvecsC, tvecsC = cv.calibrateCamera(objpoints, imgpointsC, frameSize, None, None)
heightC, widthC, channelsC = imgC.shape
newCameraMatrixC, roi_C = cv.getOptimalNewCameraMatrix(cameraMatrixC, distC, (widthC, heightC), 1, (widthC, heightC))
print(distC)
print(newCameraMatrixC)

##########################################################################################################################################################################################

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,7)
#frameSize = (1800, 2880)
frameSize = (640,480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.
imgpointsC = [] # 2d points in image plane.


#imagesLeft = sorted(glob.glob('Desktop\\test\\Kamera1\\*.png'))
imagesLeft = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/leva/*.png")))
imagesRight = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/desna/*.png")))
imagesCenter = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/center/*.png")))

for imgLeft, imgRight, imgCenter in zip(imagesLeft, imagesRight, imagesCenter):

    imgL = cv.imread(imgLeft)
    print("Left")
    print(imgL.shape)
    imgR = cv.imread(imgRight)
    print("Right")
    print(imgR.shape)
    imgC = cv.imread(imgCenter)
    print("Center")
    print(imgC.shape)

    grayL = colorToGray(imgL)
    grayR = colorToGray(imgR)
    grayC = colorToGray(imgC)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    retC, cornersC = cv.findChessboardCorners(grayC, chessboardSize, None)

    # If found, add object points, image points (after refining them)

    if not retR:
        cv.drawChessboardCorners(imgC, chessboardSize, cornersC, retC)

    if retR == True:

        diff = cornersC[chessboardSize[0]-1][0][0] - cornersC[0][0][0]
        if diff > 0:
            cornersC = cornersC[::-1].astype(np.float32)

        objpoints.append(objp)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)

cv.destroyAllWindows()



retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))
print(distR)
print(newCameraMatrixR)



############################################################################################################################################################################


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,7)
#frameSize = (1800, 2880)
frameSize = (640,480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.
imgpointsC = [] # 2d points in image plane.


#imagesLeft = sorted(glob.glob('Desktop\\test\\Kamera1\\*.png'))
imagesLeft = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/leva/*.png")))
imagesRight = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/desna/*.png")))
imagesCenter = sorted(glob.glob(os.path.expanduser("D:/prog/Python/kalib_new/center/*.png")))

for imgLeft, imgRight, imgCenter in zip(imagesLeft, imagesRight, imagesCenter):

    imgL = cv.imread(imgLeft)
    print("Left")
    print(imgL.shape)
    imgR = cv.imread(imgRight)
    print("Right")
    print(imgR.shape)
    imgC = cv.imread(imgCenter)
    print("Center")
    print(imgC.shape)

    grayL = colorToGray(imgL)
    grayR = colorToGray(imgR)
    grayC = colorToGray(imgC)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    retC, cornersC = cv.findChessboardCorners(grayC, chessboardSize, None)

    if not retL:
        cv.drawChessboardCorners(imgC, chessboardSize, cornersC, retC)

    # If found, add object points, image points (after refining them)
    if retL == True:

        diff = cornersC[chessboardSize[0]-1][0][0] - cornersC[0][0][0]
        if diff > 0:
            cornersC = cornersC[::-1].astype(np.float32)

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.waitKey(1000)

cv.destroyAllWindows()




############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L =                    cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
print(distL)
print(newCameraMatrixL)

#!/usr/bin/env python3

import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
#Calculated by hand
FOV = 55.794542
#Calculated in program
FOCAL_LENGTH = 808

#camera_matrix =np.ndarray((3,3), buffer=np.array([[517.2110395316563, 0.0, 334.3418325883858], [0.0, 543.2949569504035, 211.75050568262293], [0.0, 0.0, 1.0]])) # offset = 1*itemsize, i.e. skip first element
#dist =np.ndarray((1,5), buffer=np.array([[0.07251015712570133, -0.7601727484403693, -0.02290637861280213, 0.01562729685105215, 2.5647585221601936]])) # offset = 1*itemsize, i.e. skip first element


# Masks the video based on a range of hsv colors
# Takes in a frame, returns a masked frame
def threshold_video(frame):
    #Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    #Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    #Blurs video to smooth out image
    blur = cv2.medianBlur(frame, 5)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # define range of color in HSV
    lower_color = np.array([152, 163, 0])
    upper_color = np.array([180, 255, 255])
    # hold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Shows the theshold image in a new window
    cv2.imshow('threshold', mask)
    # Returns the masked image
    return mask

#Finds the contours from the masked image and displays them on original stream
def findContours(cap, mask):
    #Finds contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    _, frame = cap.read()
    #Flips the frame so my right is the image's right (probably need to change this
    frame = cv2.flip(frame, 1)
    #Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    #Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    #Copies frame and stores it in image
    image = frame.copy()
    #Processes the contours, takes in (contours, output_image, (centerOfImage)
    processContours(contours, image, centerX, centerY)
    #Shows the contours overlayed on the original video
    cv2.imshow("Contours", image)

#Draws and calculates contours and their properties
def processContours(contours, image, centerX, centerY):

    #Loop through all contours
    for cnt in contours:
        #Get moments of contour; mainly for centroid
        M = cv2.moments(cnt)
        #Get convex hull (bounding polygon on contour)
        hull = cv2.convexHull(cnt)
        #Calculate Contour area
        cntArea = cv2.contourArea(cnt)
        #calculate area of convex hull
        hullArea = cv2.contourArea(hull)
        #Filters contours based off of size
        if (checkContours(cntArea, hullArea)):
            #Gets the centeroids of contour
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            #Gets rotated bounding rectangle of contour
            rect = cv2.minAreaRect(cnt)
            #Creates box around that rectangle
            box = cv2.boxPoints(rect)
            #Not exactly sure
            box = np.int0(box)
            #Gets center of rotated rectangle
            center = rect[0]
            #Gets rotation of rectangle; same as rotation of contour
            rotation = rect[2]
            #Gets width and height of rotated rectangle
            width = rect[1][0]
            height = rect[1][1]
            #Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
            rotation = translateRotation(rotation, width, height)
            #Gets smaller side
            if width > height:
                smaller_side = height
            else:
                smaller_side = width
            #Calculates yaw of contour (horizontal position in degrees)
            yaw = calculateYaw(cx, centerX, FOCAL_LENGTH)
            #Calculates yaw of contour (horizontal position in degrees)
            pitch = calculatePitch(cy, centerY, FOCAL_LENGTH)
            #Adds padding for text
            padding  = -8 - math.ceil(.5*smaller_side)
            #Draws rotated rectangle
            #cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

            #Draws a vertical white line passing through center of contour
            cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
            #Draws a white circle at center of contour
            cv2.circle(image, (cx, cy), 6, (255, 255, 255))
            #Puts the rotation on screen
            cv2.putText(image, "Rotation: " + str(rotation), (cx + 40, cy + padding), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            #Puts the yaw on screen
            cv2.putText(image, "Yaw: " + str(yaw), (cx+ 40, cy + padding -16), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            #Puts the Pitch on screen
            cv2.putText(image, "Pitch: " + str(pitch), (cx+ 80, cy + padding -42), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
            #Draws the convex hull
            #cv2.drawContours(image, [hull], 0, (23, 184, 80), 3)
            #Draws the contours
            cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

            #Gets the (x, y) and radius of the enclosing circle of contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            #Rounds center of enclosing circle
            center = (int(x), int(y))
            #Rounds radius of enclosning circle
            radius = int(radius)
            #Makes bounding rectangle of contour
            rx, ry, rw, rh = cv2.boundingRect(cnt)

            #Draws countour of bounding rectangle and enclosing circle in green
            cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)
            cv2.circle(image, center, radius, (23, 184, 80), 1)



#Draws contours on blank, black image
def findContoursNewImage(cap, mask, newImage):
    # Finds contours
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    _, frame = cap.read()
    # Flips the frame so my right is the image's right (probably need to change this
    frame = cv2.flip(frame, 1)
    # Gets the shape of video
    screenHeight, screenWidth, channels = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    #makes image equal to the assigned image
    image = newImage
    #Draws contours and other stuff
    processContours(contours, image, centerX, centerY)
    #Draws on a blank image
    cv2.imshow("Contours New", image)

#Checks if contours are worthy based off of contour area and (not currently) hull area
def checkContours(cntSize, hullSize):
    if(cntSize > 1000):
        return True
    else:
        return False;


def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return rotation

#Uses trig and focal length of camera to find yaw.
#Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, focalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / focalLength))
    return yaw
#Uses trig and focal length of camera to find pitch
#Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, focalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / focalLength))
    #Just stopped working have to do this:
    pitch *= -1
    return pitch
while(True):

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    screenHeight, screenWidth, channels = frame.shape
    print("Screen width" + str(screenWidth))
    #Calculates focal Length-
    focalLength = screenWidth / (2 * math.tan(FOV/2))
    print("Focal Length " + str(focalLength))
    threshold = threshold_video(frame)
    findContours(cap, threshold)
    blank_image = np.zeros((screenHeight, screenWidth, 3), np.uint8)
    findContoursNewImage(cap, threshold, blank_image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()


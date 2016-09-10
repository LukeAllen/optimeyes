# -*- coding: utf-8 -*-
"""
Basic test of our ability to do a Haar Cascade
"""

import cv2
haarFaceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

WINDOW_NAME = "preview"

def detect(img, cascade, minimumFeatureSize=(20,20)):
    if cascade.empty():
        raise(Exception("There was a problem loading your Haar Cascade xml file."))
    #cv2.CascadeClassifier.detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) -> objects
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=minimumFeatureSize)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2] #convert last coord from (width,height) to (maxX, maxY)
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def handleFrame(frame, allowDebugDisplay=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detect(gray,haarFaceCascade)
    if allowDebugDisplay:
        output = frame
        draw_rects(output,faces,(0,255,0)) #BGR format
        cv2.imshow(WINDOW_NAME, cv2.resize(output,(0,0), fx=2,fy=2,interpolation=cv2.INTER_NEAREST) )

def main():
    previewWindow = cv2.namedWindow(WINDOW_NAME) # open a window to show debugging images

    vc = cv2.VideoCapture(0) # Initialize the default camera
    if vc.isOpened(): # try to get the first frame
        (readSuccessful, frame) = vc.read()
    else:
        print "Could not open the system camera. Is another instance already running?"
        readSuccessful = False

    while readSuccessful:
        handleFrame(frame, allowDebugDisplay=True)
        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
#            cv2.imwrite( "lastOutput.png", frame) #save the last-displayed image to file, for our report
            break
        # Get Image from camera
        readSuccessful, frame = vc.read()
    vc.release() #close the camera
    cv2.destroyWindow(WINDOW_NAME) #close the window
    
if __name__ == "__main__":
    main()

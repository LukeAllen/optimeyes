#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lukea_000
#
# Created:     11/12/2013
# Copyright:   (c) lukea_000 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import cv2
from eyeDetect import *

UPPER = 1000
UP_STEP = 1
DOWN_STEP = -1

class ClassyVirtualReferencePoint(object):
    class mykeypoint(object):
        def __init__(self, vector = (0,0), guess = (0, 0), found = 0, weight = 0):
            self.vector = vector
            self.guess = guess
            self.found = found
            self.weight = weight

    """
        Tracks a batch of keypoints, and outputs how they vote on a virtual
        reference point in an image.

        eventually will pick up new keypoints as it goes, but doesn't yet.

        by convention, all vote vectors will be of the form (referencepoint - a)
    """
    def __init__(self, keypoints, descriptors, referencePointYX, bounds, eye1, eye2):
        """
            takes keypoints and descriptors for an image, and sets up the
            internals of the class as though it's just been run.
            internals:
                knn classifier
                reference point

            the main array of descriptors is also set up, with:
                 descriptors
                 vector to virtual point
                 matchedLocation: (None if not matched)
                 weight
            as representation may change, setters and gettters should be used
        """
        self.reference = referencePointYX
        assert(len(descriptors) > 0)
        self.rowsize = len(descriptors) / len(keypoints)
        #this doesn't need to be self.saved here now, but I may want it later:
        keypoints, self.descriptors = self.cropToBounds(keypoints, descriptors, bounds, eye1, eye2)
        labels = np.arange(len(keypoints), dtype = np.int32)
        self.oldknn = cv2.ml.KNearest_create()
        #train the KNN matcher
        self.oldknn.train(self.descriptors,cv2.ml.ROW_SAMPLE, labels)
        self.keypointdata = []
        for kp in keypoints:
            vector = (self.reference[0] - kp.pt[0], self.reference[1] - kp.pt[1])
            data = self.mykeypoint(vector = vector)
            self.keypointdata.append(data)

    def drawPt(self, cx, cy, output):
        color = (255, 0, 255, 100)
        cx, cy = int(cx), int(cy)
        cv2.circle(output, (cx, cy), 5, color, thickness=2)

    def calculateReferencePoint(self):
        """
            does math for average x & y
        """
        weightedx, weightedy = (0, 0)
        denom = 0
        for kp in self.keypointdata:
            weightedx += kp.found * kp.weight * kp.guess[0]
            weightedy += kp.found * kp.weight * kp.guess[1]
            denom += kp.weight
        weightedx = weightedx / denom
        weightedy = weightedy / denom
        #set the reference point, maybe do some smoothing later
        self.reference = weightedx, weightedy

        #zero out the others if your way of setting weights changes
        for i, kp in enumerate(self.keypointdata):
            self.keypointdata[i].found = 0
        #return this
        return self.reference

    def cropToBounds(self, kps, des, bounds, eye1, eye2):
#        des = np.array(des, dtype = np.float32).reshape((-1, self.rowsize))
        keypoints, descriptors = [], []
        for i, kp in enumerate(kps):
            if containsPoint(bounds, kp.pt): #point is in the face
                if not containsPoint(eye1, kp.pt) and not containsPoint(eye2, kp.pt):
                #point is outside both eyes
                    keypoints.append(kp)
                    descriptors.append(des[i])
#        return (keypoints, np.array(descriptors, dtype = np.float32).reshape((-1, self.rowsize)))
        return (keypoints,np.array(descriptors))

    def getReferencePoint(self, keypoints, descriptors, bounds, eye1, eye2, img = None):
        """
            takes in new keypoints, and descriptors
            finds the two closest matches for each incoming one, and then verifies that they are close enough
            using D. Lowe's ratio test

            for each close enough point, it sets the weight equal to one
        """
        keypoints, descriptors = self.cropToBounds(keypoints, descriptors, bounds, eye1, eye2)
        _, _, oldneighborLabels, olddistances = self.oldknn.findNearest(descriptors, 2)
        newknn = cv2.ml.KNearest_create()
        newlabels = np.arange(len(keypoints), dtype=np.float32)
        newknn.train(descriptors, cv2.ml.ROW_SAMPLE, newlabels)
        _, _, newneighborLabels, newdistances = newknn.findNearest(self.descriptors, 2)
        #import pdb;
        for i, dists in enumerate(olddistances):
            closestDistance, other = dists
            if closestDistance < other * .7:
                #its a close match, but is it a bijection?
                #want to know whether the label of the old neighbor was matched to this new guy too
                #I know about what I have:
                #old label is the label of the nearest old point.
                oldlabel = int(oldneighborLabels[i][0])
                #want to know whether the old point at index oldlabel was labeled i, because if so, then its bijective
                if int(newneighborLabels[oldlabel][0]) ==  i:
                    newdist = newdistances[oldlabel]
                    closestDistance, other = newdist
                    if closestDistance < other * .7:
                        #really a match
                        #what index is it at?
                        self.keypointdata[oldlabel].found = 1
                        self.keypointdata[oldlabel].weight = min(UPPER, self.keypointdata[oldlabel].weight + UP_STEP)
                        #pdb.set_trace()
                        self.keypointdata[oldlabel].guess = keypoints[i].pt[0] + self.keypointdata[oldlabel].vector[0], \
                                                 keypoints[i].pt[1] + self.keypointdata[oldlabel].vector[1]
                        #if there's an image, draw on it
                        if img != None:
                            cx, cy = keypoints[i].pt#this is different from my other code but I think it's good
                            cx, cy = int(cx), int(cy)
                            cv2.putText(img, str(oldlabel), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, .25, (100, 170, 0))
        #step down the weights of the not found ones
        for kp in self.keypointdata:
            if kp.found == 0:
                kp.weight = 0
        if max([kp.weight for kp in self.keypointdata]) > 0: #if we've found at least 1 keypoint, recalculate the virtual reference
            self.calculateReferencePoint()
        if img != None:
            self.drawPt(self.reference[0], self.reference[1], img)
        return (self.reference)

def main():
    pass

if __name__ == '__main__':
    main()

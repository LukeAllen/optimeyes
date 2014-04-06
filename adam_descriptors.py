#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lukea_000
#
# Created:     15/11/2013
# Copyright:   (c) lukea_000 2013
0# Licence:     <your licence>
#-------------------------------------------------------------------------------
import cv2
import numpy
#testing open cv stuff with python
def main():
    opencv_haystack =cv2.imread('adam.jpg')
    opencv_needle = cv2.imread('adam_rightnostril.jpg')
    ngrey = cv2.cvtColor(opencv_needle, cv2.COLOR_BGR2GRAY)
    hgrey = cv2.cvtColor(opencv_haystack, cv2.COLOR_BGR2GRAY)
    import pdb
    pdb.set_trace()
    # build feature detector and descriptor extractor
    hessian_threshold = 175
    detector = cv2.SURF(hessian_threshold)
    (hkeypoints, hdescriptors) = detector.detect(hgrey, None, useProvidedKeypoints = False)
    (nkeypoints, ndescriptors) = detector.detect(ngrey, None, useProvidedKeypoints = False)

    # extract vectors of size 64 from raw descriptors numpy arrays
    rowsize = len(hdescriptors) / len(hkeypoints)
    if rowsize > 1:
        hrows = numpy.array(hdescriptors, dtype = numpy.float32).reshape((-1, rowsize))
        nrows = numpy.array(ndescriptors, dtype = numpy.float32).reshape((-1, rowsize))
        print "haystack rows shape", hrows.shape
        print "needle rows shape", nrows.shape
    else:
        print '*****************************************************8888'
        hrows = numpy.array(hdescriptors, dtype = numpy.float32)
        nrows = numpy.array(ndescriptors, dtype = numpy.float32)
        rowsize = len(hrows[0])

    # kNN training - learn mapping from hrow to hkeypoints index
    samples = hrows
    responses = numpy.arange(len(hkeypoints), dtype = numpy.float32)
    print "sample length", len(samples), "response length", len(responses)
    knn = cv2.KNearest()
    knn.train(samples,responses)

    # retrieve index and value through enumeration
    for i, descriptor in enumerate(nrows):
        descriptor = numpy.array(descriptor, dtype = numpy.float32).reshape((1, rowsize))
        print i, 'descriptor shape', descriptor.shape, 'sample shape', samples[0].shape
        retval, results, neigh_resp, dists = knn.find_nearest(descriptor, 1)
        res, dist =  int(results[0][0]), dists[0][0]
        print 'result', res, 'distance', dist

        if dist < 0.1:
            # draw matched keypoints in red color
            color = (0, 0, 255)
        else:
            # draw unmatched in blue color
            color = (255, 0, 0)
        # draw matched key points on haystack image
        x,y = hkeypoints[res].pt
        center = (int(x),int(y))
        cv2.circle(opencv_haystack,center,2,color,-1)
        # draw matched key points on needle image
        x,y = nkeypoints[i].pt
        center = (int(x),int(y))
        cv2.circle(opencv_needle,center,2,color,-1)

    cv2.imshow('haystack',opencv_haystack)
    cv2.imshow('needle',opencv_needle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import cv2
import numpy as np
import ClassyVirtualReferencePoint as ClassyVirtualReferencePoint
import ransac

# set doTraining = False to display debug graphics:
# You should do this first. There should be a green line from your
# forehead to one pupil; the end of the line is the estimate of pupil position. The blue
# circles should generally track your pupils, though less reliably than the green line.
# If performance is bad, you can tweak the "TUNABLE PARAMETER" lines. (This is a big
# area where improvement is needed; probably some learning of parameters.)
# Set True to run the main program:
# You click where you're looking, and, after around 10-20 such clicks,
# the program will learn the correspondence and start drawing a blue blur
# where you look. It's important to keep your head still (in position AND angle)
# or it won't work.
doTraining = False







def featureCenter(f):
    return (.5*(f.mExtents[0]+f.mExtents[1]),.5*(f.mExtents[2]+f.mExtents[3]) )

# returns center in form (y,x)
def featureCenterXY(rect):
    #eyes are arrays of the form [minX, minY, maxX, maxY]
    return (.5*(rect[0]+rect[2]), .5*(rect[1]+rect[3]))

def centeredBox(feature1, feature2, boxWidth, boxHeight, yOffsetToAdd = 0):
    f1 = np.array(featureCenterXY(feature1))
    f2 = np.array(featureCenterXY(feature2))
    center = (f1[:]+f2[:])/2
    center[1] += yOffsetToAdd
    offset = np.array([boxWidth/2,boxHeight/2])
    return np.concatenate( (center-offset, center+offset) )


def contains(outerFeature, innerFeature):
    p = featureCenterXY(innerFeature)
    #eyes are arrays of the form [minX, minY, maxX, maxY]
    return p[0] > outerFeature[0] and p[0] < outerFeature[2] and p[1] > outerFeature[1] and p[1] < outerFeature[3]

def containsPoint(outerFeature, p):
    #eyes are arrays of the form [minX, minY, maxX, maxY]
    return p[0] > outerFeature[0] and p[0] < outerFeature[2] and p[1] > outerFeature[1] and p[1] < outerFeature[3]

# Takes an ndarray of face rects, and an ndarray of eye rects.
# Returns the first eyes that are inside the face but not inside each other.
# Eyes are returned as the tuple (leftEye, rightEye)
def getLeftAndRightEyes(faces, eyes):
    #loop through detected faces. We'll do our processing on the first valid one.
    if len(eyes)==0:
        return ()
    for face in faces:
        for i in range(eyes.shape[0]):
            for j in range(i+1,eyes.shape[0]):
                leftEye = eyes[i] #by left I mean camera left
                rightEye = eyes[j]
                #eyes are arrays of the form [minX, minY, maxX, maxY]
                if (leftEye[0]+leftEye[2]) > (rightEye[0]+rightEye[2]): #leftCenter is > rightCenter
                    rightEye, leftEye = leftEye, rightEye #swap
                if contains(leftEye,rightEye) or contains(rightEye, leftEye):#they overlap. One eye containing another is due to a double detection; ignore it
                    debugPrint('rejecting double eye')
                    continue
                if leftEye[3] < rightEye[1] or rightEye[3] < leftEye[1]:#top of one is below (>) bottom of the other. One is likely a mouth or something, not an eye.
                    debugPrint('rejecting non-level eyes')
                    continue
##                if leftEye.minY()>face.coordinates()[1] or rightEye.minY()>face.coordinates()[1]: #top of eyes in top 1/2 of face
##                    continue;
                if not (contains(face,leftEye) and contains(face,rightEye)):#face contains the eyes. This is our standard of humanity, so capture the face.
                    debugPrint("face doesn't contain both eyes")
                    continue
                return (leftEye, rightEye)

    return ()

verbose=True

def debugPrint(s):
    if verbose:
        print(s)

showMainImg=True;

def debugImg(arr):
    global showMainImg
    showMainImg=False;
    toShow = cv2.resize((arr-arr.min())*(1.0/(arr.max()-arr.min())),(0,0), fx=8,fy=8,interpolation=cv2.INTER_NEAREST)
    cv2.imshow(WINDOW_NAME,toShow)

# displays data that is stored in a sparse format. Uses the coords to draw the corresponding
# element of the vector, on a blank image of dimensions shapeToCopy
def debugImgOfVectors(vectorToShow, gradXcoords, gradYcoords, shapeToCopy):
    img = np.zeros(shapeToCopy)
    for i,gradXcoord in enumerate(gradXcoords):
        img[gradYcoords[i]][gradXcoord] = vectorToShow[i]
    debugImg(img)

BLOWUP_FACTOR = 1 # Resizes image before doing the algorithm. Changing to 2 makes things really slow. So nevermind on this.
RELEVANT_DIST_FOR_CORNER_GRADIENTS = 8*BLOWUP_FACTOR
dilationWidth = 1+2*BLOWUP_FACTOR #must be an odd number
dilationHeight = 1+2*BLOWUP_FACTOR #must be an odd number
dilationKernel = np.ones((dilationHeight,dilationWidth),'uint8')





writeEyeDebugImages = False #enable to export image files showing pupil center probability
eyeCounter = 0

# Returns (cy,cx) of the pupil center, where y is down and x is right. You should pass in a grayscale Cv2 image which
# is closely cropped around the center of the eye (using the Haar cascade eye detector)
def getPupilCenter(gray, getRawProbabilityImage=False):
##    (scleraY, scleraX) = np.unravel_index(gray.argmax(),gray.shape)
##    scleraColor = colors[scleraY,scleraX,:]
##    img[scleraX,scleraY] = (255,0,0)
##    img.colorDistance(skinColor[:]).save(disp)
##    img.edges().save(disp)
##    print skinColor, scleraColor
    gray = gray.astype('float32')
    if BLOWUP_FACTOR != 1:
        gray = cv2.resize(gray, (0,0), fx=BLOWUP_FACTOR, fy=BLOWUP_FACTOR, interpolation=cv2.INTER_LINEAR)

    IRIS_RADIUS = gray.shape[0]*.75/2 #conservative-large estimate of iris radius TODO: make this a tracked parameter--pass a prior-probability of radius based on last few iris detections. TUNABLE PARAMETER
    #debugImg(gray)
    dxn = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3) #optimization opportunity: blur the image once, then just subtract 2 pixels in x and 2 in y. Should be equivalent.
    dyn = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitudeSquared = np.square(dxn)+np.square(dyn)

    # ########### Pupil finding
    magThreshold = magnitudeSquared.mean()*.6 #only retain high-magnitude gradients. <-- VITAL TUNABLE PARAMETER
                    # The value of this threshold is critical for good performance.
                    # todo: adjust this threshold using more images. Maybe should train our tuned parameters.
    # form a bool array, unrolled columnwise, which can index into the image.
    # we will only use gradients whose magnitude is above the threshold, and
    # (optionally) where the gradient direction meets characteristics such as being more horizontal than vertical.
    gradsTouse = (magnitudeSquared>magThreshold) & (np.abs(4*dxn)>np.abs(dyn))
    lengths = np.sqrt(magnitudeSquared[gradsTouse]) #this converts us to double format
    gradDX = np.divide(dxn[gradsTouse],lengths) #unrolled columnwise
    gradDY = np.divide(dyn[gradsTouse],lengths)
##    debugImg(gradsTouse*255)
##    ksize = 7 #kernel size = x width and y height of the filter
##    sigma = 4
##    blurredGray = cv2.GaussianBlur(gray, (ksize,ksize), sigma, borderType=cv2.BORDER_REPLICATE)
##    debugImg(gray)
##    blurredGray = cv2.blur(gray, (ksize,ksize)) #x width and y height. TODO: try alternately growing and eroding black instead of blurring?
    #isDark = blurredGray < blurredGray.mean()
    isDark = gray< (gray.mean()*.8)  #<-- TUNABLE PARAMETER
    global dilationKernel
    isDark = cv2.dilate(isDark.astype('uint8'), dilationKernel) #dilate so reflection goes dark too
##    isDark = cv2.erode(isDark.astype('uint8'), dilationKernel)
##    debugImg(isDark*255)
    gradXcoords =np.tile( np.arange(dxn.shape[1]), [dxn.shape[0], 1])[gradsTouse] # build arrays holding the original x,y position of each gradient in the list.
    gradYcoords =np.tile( np.arange(dxn.shape[0]), [dxn.shape[1], 1]).T[gradsTouse] # These lines are probably an optimization target for later.
    minXForPupil = 0 #int(dxn.shape[1]*.3)
##    #original method
##    centers = np.array([[phi(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords) if isDark[cy][cx] else 0 for cx in range(dxn.shape[1])] for cy in range(dxn.shape[0])])
    #histogram method
    centers = np.array([[phiWithHist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS) if isDark[cy][cx] else 0 for cx in xrange(minXForPupil,dxn.shape[1])] for cy in xrange(dxn.shape[0])]).astype('float32')
    # display outputs for debugging
##    centers = np.array([[phiTest(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords) for cx in range(dxn.shape[1])] for cy in range(dxn.shape[0])])
##    debugImg(centers)
    maxInd = centers.argmax()
    (pupilCy,pupilCx) = np.unravel_index(maxInd, centers.shape)
    pupilCx += minXForPupil
    pupilCy /= BLOWUP_FACTOR
    pupilCx /= BLOWUP_FACTOR
    if writeEyeDebugImages:
        global eyeCounter
        eyeCounter = (eyeCounter+1)%5 #write debug image every 5th frame
        if eyeCounter == 1:
            cv2.imwrite( "eyeGray.png", gray/gray.max()*255) #write probability images for our report
            cv2.imwrite( "eyeIsDark.png", isDark*255)
            cv2.imwrite( "eyeCenters.png", centers/centers.max()*255)
    if getRawProbabilityImage:
        return (pupilCy, pupilCx, centers)
    else:
        return (pupilCy, pupilCx)


lastCornerProb = np.ones([1,1])

#This was a failed attempt to find eye corners, not used in final version.
# Returns (cy,cx) of the eye corner, where y is down and x is right. You should pass in a grayscale Cv2 image which
# is closely cropped around the corner of the eye (using the Haar cascade eye detector)
def getEyeCorner(gray):
##    (scleraY, scleraX) = np.unravel_index(gray.argmax(),gray.shape)
##    scleraColor = colors[scleraY,scleraX,:]
##    img[scleraX,scleraY] = (255,0,0)
##    img.colorDistance(skinColor[:]).save(disp)
##    img.edges().save(disp)
##    print skinColor, scleraColor

    if BLOWUP_FACTOR != 1:
        gray = cv2.resize(gray, (0,0), fx=BLOWUP_FACTOR, fy=BLOWUP_FACTOR, interpolation=cv2.INTER_LINEAR)
    gray = gray.astype('float32')
    #debugImg(gray)
    dxn = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3) #optimization opportunity: blur the image once, then just subtract 2 pixels in x and 2 in y. Should be equivalent.
    dyn = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    magnitudeSquared = np.square(dxn)+np.square(dyn)
##    debugImg(np.sqrt(magnitudeSquared))
    # ########### Eye corner finding. TODO: limit gradients to search area
    rangeOfXForCorner = int(dxn.shape[1]/2)
    magThreshold = magnitudeSquared.mean()*.5 #only retain high-magnitude gradients. todo: adjust this threshold using more images. Maybe should train our tuned parameters.
    # form a bool array, unrolled columnwise, which can index into the image.
    # we will only use gradients whose magnitude is above the threshold, and
    # (optionally) where the gradient direction meets characteristics such as being more horizontal than vertical.
    gradsTouse = (magnitudeSquared>magThreshold) & (np.abs(2*dyn)>np.abs(dxn))
    lengths = np.sqrt(magnitudeSquared[gradsTouse])
    gradDX = np.divide(dxn[gradsTouse],lengths) #unrolled columnwise
    gradDY = np.divide(dyn[gradsTouse],lengths)
    gradXcoords =np.tile( np.arange(dxn.shape[1]), [dxn.shape[0], 1])[gradsTouse] # build arrays holding the original x,y position of each gradient in the list.
    gradYcoords =np.tile( np.arange(dxn.shape[0]), [dxn.shape[1], 1]).T[gradsTouse] # These lines are probably an optimization target for later.
##    debugImgOfVectors(gradDY,gradXcoords,gradYcoords, dxn.shape)
    centers = np.array([[phiCorner(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords) for cx in range(rangeOfXForCorner)] for cy in range(dxn.shape[0])])
##    debugImg(centers)
    # Tracking -- use prior beliefs about corner position
    global lastCornerProb
    weightOnNew = 1
    prior = np.ones(centers.shape)*(lastCornerProb.mean()*.5)*(1-weightOnNew) # fill with default value
    startPrior = [0,0]
    endPrior = [0,0]
    startNew = [0,0]
    endNew = [0,0]
    for i in xrange(2):
        diff = lastCornerProb.shape[i]-centers.shape[i]
        if diff >= 0: # new is smaller
            startPrior[i] = int(diff/2)
            endPrior[i] = startPrior[i]+centers.shape[i]
            startNew[i]=0
            endNew[i]=centers.shape[i]
        else: # prior is smaller
            startPrior[i] = 0
            endPrior[i] = lastCornerProb.shape[i]
            startNew[i]=int(-diff/2)
            endNew[i]=startNew[i]+lastCornerProb.shape[i]
    prior[startNew[0]:endNew[0]][startNew[1]:endNew[1]] = (1-weightOnNew)*lastCornerProb[startPrior[0]:endPrior[0]][startPrior[1]:endPrior[1]]
    centers = centers * (1/centers.max())
    centers = centers * (weightOnNew + prior)
##    debugImg(centers)
    (cornerCy,cornerCx) = np.unravel_index(centers.argmax(), centers.shape)
    cornerCy /= BLOWUP_FACTOR
    cornerCx /= BLOWUP_FACTOR
    return (cornerCy, cornerCx)

# Estimates the probability that the given cx,cy is the pupil center, by taking
# (its vector to each gradient location) dot (the gradient vector)
def phi(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS):
    vecx = gradXcoords-cx
    vecy = gradYcoords-cy
    lengthsSquared = np.square(vecx)+np.square(vecy)
    valid = (lengthsSquared > 0) & (lengthsSquared < IRIS_RADIUS**2) #avoid divide by zero, only use nearby gradients.
    dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
    valid = valid & (dotProd > 0) # only use vectors in the same direction (i.e. the dark-to-light transition direction is away from us. The good gradients look like that.)
    dotProd = np.square(dotProd[valid]) # dot products squared
    dotProd = np.divide(dotProd,lengthsSquared[valid]) #normalized squared dot products. Should range from 0 to 1.
##    import pdb;pdb.set_trace()
    dotProd = dotProd[dotProd > .9] #only count dot products that are really close
    return np.sum(dotProd) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2

# Estimates the probability that the given cx,cy is the pupil center, by taking
# (its vector to each gradient location) dot (the gradient vector)
# only uses gradients which are near the peak of a histogram of distance
# cx and cy may be integers or floating point.
def phiWithHist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS):
    vecx = gradXcoords-cx
    vecy = gradYcoords-cy
    lengthsSquared = np.square(vecx)+np.square(vecy)
    # bin the distances between 1 and IRIS_RADIUS. We'll discard all others.
    binWidth = 1 #TODO: account for webcam resolution. Also, maybe have it transform ellipses to circles when on the sides? (hard)
    numBins =  int(np.ceil((IRIS_RADIUS-1)/binWidth))
    bins = [(1+binWidth*index)**2 for index in xrange(numBins+1)] #express bin edges in terms of length squared
    hist = np.histogram(lengthsSquared, bins)[0]
    maxBin = hist.argmax()
    slop = binWidth
    valid = (lengthsSquared > max(1,bins[maxBin]-slop)) &  (lengthsSquared < bins[maxBin+1]+slop) #use only points near the histogram distance
    dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
    valid = valid & (dotProd > 0) # only use vectors in the same direction (i.e. the dark-to-light transition direction is away from us. The good gradients look like that.)
    dotProd = np.square(dotProd[valid]) # dot products squared
    dotProd = np.divide(dotProd,lengthsSquared[valid]) #make normalized squared dot products
##    dotProd = dotProd[dotProd > .9] #only count dot products that are really close
    dotProd = np.square(dotProd) # squaring puts an even higher weight on values close to 1
    return np.sum(dotProd) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2

# Failed attempt to find probability that the given cx,cy is an eye corner, not
# used in final version. Works by taking
# (its vector to each gradient location) dot (the gradient vector). The corner
# should have a near-zero dot product with its nearby vectors, because the
# eyelids form lines that point right at the eye corner (so their gradients are
# at a 90 degree angle to it).
# only uses gradients which are near the peak of a histogram of distance
# cx and cy may be integers or floating point.
def phiCorner(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords):
    vecx = gradXcoords-cx
    vecy = gradYcoords-cy
    angles = np.arctan2(vecy,vecx)
    lengthsSquared = np.square(vecx)+np.square(vecy)
    valid = (lengthsSquared > 0) & (lengthsSquared < RELEVANT_DIST_FOR_CORNER_GRADIENTS) & (vecx>0.4)#RIGHT EYE ASSUMPTION
    numBins = 10
##    binWidth = 2.0/numBins
##    slop = binWidth/2
##    bins = [binWidth*i for i in xrange(numBins+1)]
##    hist = np.histogram(gradDY[valid], bins)[0]
    (hist,bins) = np.histogram(angles, numBins, (-math.pi,math.pi))
    slop = math.pi/numBins/2
    maxBin = hist.argmax()
    hist[maxBin] = 0;
    hist[max(0,maxBin-1)]=0;
    hist[min(maxBin+1,numBins-1)]=0;
    secondMaxBin = hist.argmax();
    stat = angles #gradDY
    validBina = valid & ((bins[maxBin]-slop<stat)&(stat<bins[maxBin+1]+slop))
    validBinb = valid & ((bins[secondMaxBin]-slop<stat)&(stat<bins[secondMaxBin+1]+slop))#use only points near the histogram max


    dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
    dotProda = np.square(dotProd[validBina]) # dot products squared
    dotProdb = np.square(dotProd[validBinb]) # dot products squared
    dotProda = 1.0-np.divide(dotProda,lengthsSquared[validBina]) #make normalized squared dot products, and take 1-them so 0 gets the highest score
    dotProdb = 1.0-np.divide(dotProdb,lengthsSquared[validBinb]) #make normalized squared dot products, and take 1-them so 0 gets the highest score
##    import pdb;pdb.set_trace()
    dotProda = np.square(dotProda) #only count dot products that are really close
    dotProdb = np.square(dotProdb) #only count dot products that are really close
    suma = np.sum(dotProda) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2
    sumb = np.sum(dotProdb) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2
    return min(suma,sumb)+.5*max(suma,sumb) #this score should favor a strong bimodal histogram shape

#for debugging
def phiTest(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords):
    for ix,xcoord in enumerate(gradXcoords):
        if xcoord==cx and gradYcoords[ix]==cy:
            return np.atan2(gradDY[ix],gradDX[ix])
    return 0

# multiplies newProb and priorToMultiply
# YXoffsetOfSecondWithinFirst - priorToMultiply will be shifted by this amount in space
# defaultPriorValue - if not all of newProb is covered by priorToMultiply, this scalar goes in the uncovered areas.
def multiplyProbImages(newProb, priorToMultiply, YXoffsetOfSecondWithinFirst, defaultPriorValue):
    if np.any(YXoffsetOfSecondWithinFirst > newProb.shape) or np.any(-YXoffsetOfSecondWithinFirst > priorToMultiply.shape):
        print "multiplyProbImages aborting - zero overlap. Offset and matrices:"
        print YXoffsetOfSecondWithinFirst
        print newProb.shape
        print priorToMultiply.shape
        return newProb*defaultPriorValue
    prior = np.ones(newProb.shape)*defaultPriorValue # Most of this will get overwritten. For areas that won't be, with fill with default value.
    #offsets
    startPrior = [0,0]
    endPrior = [0,0]
    startNew = [0,0]
    endNew = [0,0]
    for i in xrange(2):
        #offset=0
        # NOT THIS: x[1:2][1:2]
        # THIS: x[1:2,1:2]
        offset = int(round(YXoffsetOfSecondWithinFirst[i])) # how much to offset 'prior' within 'newProb', for the current dimension
        print offset
        if offset >= 0: # prior goes right of 'newProb', in the world. So prior will be copied into newProb at a positive offset
            startPrior[i] = 0 #index within prior
            endPrior[i] = min(priorToMultiply.shape[i],newProb.shape[i]-offset) #how much of prior to copy
            startNew[i]=offset
            endNew[i]=offset+endPrior[i]
        else: # prior goes left of 'newProb', in the world.
            startPrior[i] = -offset
            endPrior[i] = min(priorToMultiply.shape[i], startPrior[i]+newProb.shape[i])
            startNew[i]=0
            endNew[i]=endPrior[i]-startPrior[i]
    prior[startNew[0]:endNew[0],startNew[1]:endNew[1]] = priorToMultiply[startPrior[0]:endPrior[0],startPrior[1]:endPrior[1]]
    #prior[1:10,1:10] = priorToMultiply[1:10,1:10]
    #now, prior holds the portion of priorToMultiply which overlapped newProb.
    return newProb * prior


## img: cv2 image in uint8 format
## cascade: object you made with cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
## minimumFeatureSize (ySize,xSize) tuple holding the smallest object you'd be looking for. E.g. (30,30)
## returns a numpy ndarray where rects[0] is the first detection, and holds [minX, minY, maxX, maxY] where +Y = downward
def detect(img, cascade, minimumFeatureSize=(20,20)):
    if cascade.empty():
        raise(Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2] #convert last coord from (width,height) to (maxX, maxY)
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)




# init the filters we'll use below
haarFaceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
haarEyeCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye.xml")
#img.listHaarFeatures() displays these Haar options:
#['eye.xml', 'face.xml', 'face2.xml', 'face3.xml', 'face4.xml', 'fullbody.xml', 'glasses.xml', 'lefteye.xml', #'left_ear.xml', 'left_eye2.xml', 'lower_body.xml', 'mouth.xml', 'nose.xml', 'profile.xml',
#'right_ear.xml', 'right_eye.xml', 'right_eye2.xml', 'two_eyes_big.xml', 'two_eyes_small.xml', 'upper_body.xml', #'upper_body2.xml']
OffsetRunningAvg = None
PupilSpacingRunningAvg = None

# global stuff for Adam's virtual ref point
#initialize the SURF descriptor
hessianThreshold = 500
nOctaves = 4
nOctaveLayers = 2
extended = True
upright = True
detector = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
#figure out a way to nearest neighbor map to index
virtualpoint = None
warm=0

#*********  getOffset  **********
# INPUTS:
# frame - a color numpy image.
# allowDebugDisplay - pass True if you want it to draw pupil centers, etc on "frame" and then display it.
# Display requires that you called this line to create the window: previewWindow = cv2.namedWindow(WINDOW_NAME)
# trackAverageOffset - output will be a moving average rather than instantaneous value
# directInferenceLeftRight - combines probability images from left and right to hopefully reduce noise in estimation of pupil offset
# Returns a list of two tuples of pupil offsets from the forehead dot. Specifically:
# [(cameraLeftEyeOffsetX, cameraLeftEyeOffsetY),  (cameraRightEyeOffsetX, cameraRightEyeOffsetY) ]
# If no valid face is found, returns None.
# Requires the functions above.
def getOffset(frame, allowDebugDisplay=True, trackAverageOffset=True, directInferenceLeftRight=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # find faces and eyes
    minFaceSize = (80,80)
    minEyeSize = (25,25)
    faces = detect(gray,haarFaceCascade,minFaceSize)
    eyes = detect(gray,haarEyeCascade,minEyeSize)
    drawKeypoints = allowDebugDisplay #can set this false if you don't want the keypoint ID numbers
    if allowDebugDisplay:
        output = frame
        draw_rects(output,faces,(0,255,0)) #BGR format
    else:
        output = None
##        draw_rects(output,eyes,(255,0,0))
    leftEye_rightEye = getLeftAndRightEyes( faces, eyes)
    if leftEye_rightEye: #if we found valid eyes in a face
##            draw_rects(output,leftEye_rightEye,(0,0,255)) #BGR format
        xDistBetweenEyes = (leftEye_rightEye[0][0]+leftEye_rightEye[0][1]+leftEye_rightEye[1][0]+leftEye_rightEye[1][1])/4 #for debugging reference point
        pupilXYList = []
        pupilCenterEstimates = []
        for eyeIndex, eye in enumerate(leftEye_rightEye):
##        eye = leftEye_rightEye[1]
            corner = eye.copy()

            #eyes are arrays of the form [minX, minY, maxX, maxY]
            eyeWidth = eye[2]-eye[0]
            eyeHeight = eye[3]-eye[1]
            eye[0] += eyeWidth*.20
            eye[2] -= eyeWidth*.15
            eye[1] += eyeHeight*.3
            eye[3] -= eyeHeight*.2
            eye = np.round(eye)
            eyeImg = gray[eye[1]:eye[3], eye[0]:eye[2]]
            if directInferenceLeftRight:
                (cy,cx, centerProb) = getPupilCenter(eyeImg, True)
                pupilCenterEstimates.append(centerProb.copy())
            else:
                (cy,cx) = getPupilCenter(eyeImg, True)
            pupilXYList.append( (cx+eye[0],cy+eye[1])  )
            if allowDebugDisplay:
                cv2.rectangle(output, (eye[0], eye[1]), (eye[2], eye[3]), (0,255,0), 1)
                cv2.circle(output, pupilXYList[eyeIndex], 3, (255,0,0),thickness=1) #BGR format


        # tear-duct of the camera-right eye
##        corner[0] += eyeWidth*0
##        corner[2] -= eyeWidth*.6
##        corner[1] += eyeHeight*.4
##        corner[3] -= eyeHeight*.35
##        corner = np.round(corner)
##        cv2.rectangle(output, (corner[0], corner[1]), (corner[2], corner[3]), (0,0,0), 1)
##        cornerImg = gray[corner[1]:corner[3], corner[0]:corner[2]]
##        (cornerCy,cornerCx) = getEyeCorner(cornerImg)
##        cv2.circle(output, (cornerCx+corner[0],cornerCy+corner[1]), 2, (255,255,0),thickness=1) #BGR format

        # direct inference combination of the two eye probability images.
        global PupilSpacingRunningAvg
        if directInferenceLeftRight:
            # these vectors are in XY format
            pupilSpacing = np.array(pupilXYList[1])-np.array(pupilXYList[0]) # vector from pupil 0 to pupil 1
            if PupilSpacingRunningAvg is None:
                PupilSpacingRunningAvg = pupilSpacing
            else:
                weightOnNew = .03
                PupilSpacingRunningAvg = (1-weightOnNew)*PupilSpacingRunningAvg + weightOnNew*pupilSpacing  # vector from pupil 0 to pupil 1
            if allowDebugDisplay:
                cv2.line(output, (int(pupilXYList[0][0]),int(pupilXYList[0][1])), (int(pupilXYList[0][0]+PupilSpacingRunningAvg[0]), int(pupilXYList[0][1]+PupilSpacingRunningAvg[1])), (0,100,100))
            imageZeroToOneVector = leftEye_rightEye[1][0:2]-leftEye_rightEye[0][0:2] # vector from eyeImg 0 to 1
            positionOfZeroWithinOne = PupilSpacingRunningAvg-imageZeroToOneVector; # the extra distance that wasn't covered by the bounding boxes should be applied as an offset when multiplying images.
            ksize = 5 #kernel size = x width and y height of the filter
            sigma = 2
            for i,centerEstimate in enumerate(pupilCenterEstimates):
                pupilCenterEstimates[i] = cv2.GaussianBlur(pupilCenterEstimates[i], (ksize,ksize), sigma, borderType=cv2.BORDER_REPLICATE)
            jointPupilProb = multiplyProbImages(pupilCenterEstimates[1], pupilCenterEstimates[0], positionOfZeroWithinOne[::-1], 0) # the [::-1] reverse the order, so it's YX instead of the XY that these vectors are in
##            debugImg(jointPupilProb)
            maxInd = jointPupilProb.argmax()
##            cv2.imwrite( "eye0.png", pupilCenterEstimates[0]/pupilCenterEstimates[0].max()*255) #write probability images for our report
##            cv2.imwrite( "eye1.png", pupilCenterEstimates[1]/pupilCenterEstimates[1].max()*255)
##            cv2.imwrite( "eyeJoint.png", jointPupilProb/jointPupilProb.max()*255)
            (pupilCy,pupilCx) = np.unravel_index(maxInd, jointPupilProb.shape) # coordinates in the eye 1 (camera-right eye) image
            pupilXYList[0]=pupilXYList[1]=(pupilCx + leftEye_rightEye[1][0],pupilCy + leftEye_rightEye[1][1]) #convert to absolute image coordinates


        useSURFReference = True
        if not useSURFReference: # this code assumes you have drawn a dark dot on your forehead. Should be drawn between the eyes, about the size of the iris.
            dotSearchBox = np.round( centeredBox(leftEye_rightEye[0], leftEye_rightEye[1], xDistBetweenEyes*.2, xDistBetweenEyes*.3, -xDistBetweenEyes*.09 ) ).astype('int')

            (refY,refX) = getPupilCenter(gray[dotSearchBox[1]:dotSearchBox[3], dotSearchBox[0]:dotSearchBox[2]])
            refXY = (refX+dotSearchBox[0],refY+dotSearchBox[1])
            if allowDebugDisplay:
                cv2.rectangle(output, (dotSearchBox[0], dotSearchBox[1]), (dotSearchBox[2], dotSearchBox[3]), (128,0,128), 1)
                cv2.circle(output, refXY, 2, (0,0,100),thickness=1) #BGR format
        else: # Adam's virtual reference point code. See paper for how it works.
            refXY = (0,0)
            global warm, virtualpoint
            warm += 1
            if warm > 8:
                #adam
                face = faces[0]#expect the first one
                faceImg = gray[face[1]:face[3], face[0]:face[2]]
                cornerImg = gray[corner[1]:corner[3], corner[0]:corner[2]]
                if virtualpoint == None: #we haven't set up the reference point yet
                    haystackKeypoints, haystackDescriptors = detector.detectAndCompute(gray, mask=None)
                    if len(haystackKeypoints) != 0:
                        betweenEyes = (np.array(featureCenterXY(leftEye_rightEye[0]))+np.array(featureCenterXY(leftEye_rightEye[1])))/2
                        virtualpoint = ClassyVirtualReferencePoint.ClassyVirtualReferencePoint(haystackKeypoints, haystackDescriptors, (betweenEyes[0], betweenEyes[1]), face, leftEye_rightEye[0], leftEye_rightEye[1])
                    else:
                        print "begin fail"
                else: #we've already created it
                    keypoints, descriptors = detector.detectAndCompute(gray, mask=None)
                    if drawKeypoints:
                        imgToDrawOn = output
                    else:
                        imgToDrawOn = None
                    if len(descriptors) != 0:
                        refXY  = virtualpoint.getReferencePoint(keypoints, descriptors, face, leftEye_rightEye[0], leftEye_rightEye[1], imgToDrawOn)
            # end of Adam's reference point code

        for i in xrange(len(pupilXYList)):
            pupilXYList[i] = ( pupilXYList[i][0]-refXY[0], pupilXYList[i][1]-refXY[1])
        pupilXYList = list(pupilXYList[0])+ list(pupilXYList[1]) #concatenate cam-left and cam-right coordinate tuples to make a single length 4 vector [x,y,x,y]

        if trackAverageOffset: # this frame's estimated offset will be a weighted average of the new measurement and the last frame's estimated offset
            global OffsetRunningAvg
            if OffsetRunningAvg is None:
                OffsetRunningAvg = np.array( [0,0])
            weightOnNew = .4; #Tuned parameter, must be >0 and <=1.0. Increase for faster response, decrease for better noise rejection.
            currentOffset = (np.array(pupilXYList[:2])+np.array(pupilXYList[2:]))/2
            OffsetRunningAvg = (1.0-weightOnNew)*OffsetRunningAvg + weightOnNew*currentOffset
            pupilXYList = OffsetRunningAvg
##            import pdb; pdb.set_trace()
            if allowDebugDisplay:
                cv2.line(output, (int(refXY[0]),int(refXY[1])), (int(refXY[0]+pupilXYList[0]), int(refXY[1]+pupilXYList[1])), (0,255,100))

        if allowDebugDisplay and showMainImg:
            # Double size
            cv2.imshow(WINDOW_NAME, cv2.resize(output,(0,0), fx=2,fy=2,interpolation=cv2.INTER_NEAREST) )
            # original size

        return tuple(pupilXYList) # if trackAverageOffset, it's length 2 and holds the average offset. Else, it's length 4 (old code)

    else: # no valid face was found
        if allowDebugDisplay:
            cv2.imshow(WINDOW_NAME, cv2.resize(output,(0,0), fx=2,fy=2,interpolation=cv2.INTER_NEAREST) )
        return None


class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class fulfills the model interface needed by the ransac() function.

    """
    # lists of indices of input and output columns
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
##        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
##        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
##        x,resids,rank,s = scipy.linalg.lstsq(A,B)
##        return x
        HT = np.linalg.lstsq(data[:,self.input_columns], data[:,self.output_columns])[0] # returns a tuple, where index 0 is the solution matrix.
        return HT

    def get_error( self, data, model):
        B_fit = data[:,self.input_columns].dot(model)
        err_per_point = np.sum((data[:,self.output_columns]-B_fit)**2,axis=1) # sum squared error per row
        err_per_point = np.sqrt(err_per_point) # I'll see if this helps. If not remove for speed.
        return err_per_point

def getFeatures(XYOffsets, quadratic = True):
##    print XYOffsets
    if len(XYOffsets.shape)==1:
        numRows=1
        XYOffsets.shape = (numRows,XYOffsets.shape[0])
    else:
        numRows =XYOffsets.shape[0]
    numCols = XYOffsets.shape[1]

    data = np.concatenate( (XYOffsets, np.ones( (XYOffsets.shape[0],1)) ) , axis=1) # [x,y,1]
    if quadratic:
        squaredFeatures = np.square(XYOffsets)
        squaredFeatures.shape = (numRows,numCols)
        xy = XYOffsets[:,0]*XYOffsets[:,1]
        xy.shape = (numRows,1)
##        print(xy.shape)

        data = np.concatenate( (data,squaredFeatures, xy ) , axis=1) # [x,y,1,x^2,y^2,xy]
    return data


RANSAC_MIN_INLIERS = 7
def RANSACFitTransformation(OffsetsAndPixels):
    numInputCols = OffsetsAndPixels.shape[1]-2
    data = np.concatenate( (OffsetsAndPixels[:,0:numInputCols], OffsetsAndPixels[:,numInputCols:] ) , axis=1)

    model = LinearLeastSquaresModel(range(numInputCols), (numInputCols,numInputCols+1))
    minSeedSize = 5
    iterations = 800
    maxInlierError = 240 #**2
    HT = ransac.ransac(data, model, minSeedSize, iterations, maxInlierError, RANSAC_MIN_INLIERS)
    return HT

def fitTransformation(OffsetsAndPixels):
    offsets = np.concatenate( (OffsetsAndPixels[:,0:2], np.ones( (OffsetsAndPixels.shape[0],1)) ) , axis=1)
    pixels = OffsetsAndPixels[:,2:]
    HT = np.linalg.lstsq(offsets, pixels)[0] # returns a tuple, where index 0 is the solution matrix.
    return HT

WINDOW_NAME = "preview"
def main():
    cv2.namedWindow(WINDOW_NAME) # open a window to show debugging images
    vc = cv2.VideoCapture(0) # Initialize the default camera
    try:
        if vc.isOpened(): # try to get the first frame
            (readSuccessful, frame) = vc.read()
        else:
            raise(Exception("failed to open camera."))
            readSuccessful = False
    
        while readSuccessful:
            pupilOffsetXYList = getOffset(frame, allowDebugDisplay=True)
            key = cv2.waitKey(10)
            if key == 27: # exit on ESC
                cv2.imwrite( "lastOutput.png", frame) #save the last-displayed image to file, for our report
                break
            # Get Image from camera
            readSuccessful, frame = vc.read()
    finally:
        vc.release() #close the camera
        cv2.destroyWindow(WINDOW_NAME) #close the window

def mainForTraining():
    import pygamestuff
    crosshair = pygamestuff.Crosshair([7, 2], quadratic = False)
    vc = cv2.VideoCapture(0) # Initialize the default camera
    if vc.isOpened(): # try to get the first frame
        (readSuccessful, frame) = vc.read()
    else:
        raise(Exception("failed to open camera."))
        return

    MAX_SAMPLES_TO_RECORD = 999999
    recordedEvents=0
    HT = None
    try:
        while readSuccessful and recordedEvents < MAX_SAMPLES_TO_RECORD and not crosshair.userWantsToQuit:
            pupilOffsetXYList = getOffset(frame, allowDebugDisplay=False)
            if pupilOffsetXYList is not None: #If we got eyes, check for a click. Else, wait until we do.
                if crosshair.pollForClick():
                    crosshair.clearEvents()
                    #print( (xOffset,yOffset) )
                    #do learning here, to relate xOffset and yOffset to screenX,screenY
                    crosshair.record(pupilOffsetXYList)
                    print "recorded something"
                    crosshair.remove()
                    recordedEvents += 1
                    if recordedEvents > RANSAC_MIN_INLIERS:
    ##                    HT = fitTransformation(np.array(crosshair.result))
                        resultXYpxpy =np.array(crosshair.result)
                        features = getFeatures(resultXYpxpy[:,:-2])
                        featuresAndLabels = np.concatenate( (features, resultXYpxpy[:,-2:] ) , axis=1)
                        HT = RANSACFitTransformation(featuresAndLabels)
                        print HT
                if HT is not None: # draw predicted eye position
                    currentFeatures =getFeatures( np.array( (pupilOffsetXYList[0], pupilOffsetXYList[1]) ))
                    gazeCoords = currentFeatures.dot(HT)
                    crosshair.drawCrossAt( (gazeCoords[0,0], gazeCoords[0,1]) )
            readSuccessful, frame = vc.read()
    
        print "writing"
        crosshair.write() #writes data to a csv for MATLAB
        crosshair.close()
        print "HT:\n"
        print HT
        resultXYpxpy =np.array(crosshair.result)
        print "eyeData:\n"
        print getFeatures(resultXYpxpy[:,:-2])
        print "resultXYpxpy:\n"
        print resultXYpxpy[:,-2:]
    finally:
        vc.release() #close the camera
    




if __name__ == '__main__':
    if doTraining:
        mainForTraining()
    else:
        main()
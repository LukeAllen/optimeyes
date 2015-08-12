Read the Optimeyes Theory Paper above to see principles of operation. The major advances relative to other code are:

1. The virtual reference point, which uses multiple unreliable keypoints to derive a very reliable reference point on the face.
2. The method of overlaying one eye's pupil-probability image on the other, to greatly increase confidence of the estimate.

To use the project on your computer:

- Option 1 (easiest setup): use the "master" branch of this repo, and install SimpleCV 1.3 from simplecv.org. Ensure your OpenCV version is 2.x (version 3.x has different function signatures)
- Option 2 (slightly higher performance): use the "opencv3" branch of this repo, and install OpenCV 3.0 from opencv.org.
- run eyeDetect.py. 

If the "doTraining" variable at the top of eyeDetect.py is False, it will display pupil centers graphically as shown in the report. Do that first, to verify lighting conditions, webcam field of view, etc. 

If "doTraining" is true, it will produce a Pygame window. Gaze at the mouse cursor for 1 second to let things stabilize, click the mouse, and repeat at a new position (for good results you must wait 1 second before clicking, every time). When enough training data has been collected to yield a good fit (typically 10 to 30 clicks), a blue blur will appear centered at the predicted eye gaze position. You can do more clicks to further improve the fit. Be sure to keep your head as stationary as possible the whole time. (Don't move OR rotate your head at all.)

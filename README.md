# Optimeyes
A demo of pupil tracking using a normal webcam. *Note: this is just a proof of concept, not a production library. Contributors are welcome though.*

Read the Optimeyes Theory Paper above to see principles of operation. The major advances relative to other code are:

1. The virtual reference point, which uses multiple unreliable keypoints to derive a very reliable reference point on the face.
2. The method of overlaying one eye's pupil-probability image on the other, to greatly increase confidence of the estimate.

## Install
```
pip install -r requirements.txt
```

### Instal OpenCV
Build from source is required for using opencv with contrib, non-free modules (SURF).

#### Windows
Build and install opencv-python with contrib, non-free modules wheel
```
git clone --recursive https://github.com/opencv/opencv-python.git
cd opencv-python
SET CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON"
SET ENABLE_CONTRIB=1
python setup.py bdist_wheel
pip install dist\opencv_contrib_python-x.x.x.x.whl
```
*Note: change the name of the wheel to the wheel built in the dist of opencv-python*

#### Installation on Linux
Install dependencies
```bash
apt-get update && apt-get install -y --no-install-recommends build-essential python-dev cmake git pkg-config libjpeg8-dev libjasper-dev libpng12-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev gfortran
```

Build opencv-python with contrib, non-free modules
```bash
git clone --recursive https://github.com/opencv/opencv-python.git
cd opencv-python
export CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON"
export ENABLE_CONTRIB=1
python setup.py bdist_wheel
pip install dist/opencv_contrib_python-x.x.x.x.whl
```
*Note: change the name of the wheel to the wheel built in the dist of opencv-python*

## Run
```
python eyeDetect.py
```

On your first run, ensure the "doTraining" variable at the top of eyeDetect.py is False. This makes it display pupil centers graphically, as depicted in the Theory Paper. The green line from the virtual reference point to your pupil should be stable and should track your eye movement. It helps to get as close to the camera as you can-- your eyes should be >30 pixels tall. And your face should be well-lit, so that your pupils are clearly visible. (Note: you'll need to restart the program every time you move your head or change lighting conditions.)

When the pupil tracking looks good, set "doTraining" to True and run again. It will produce a Pygame window. To train the gaze detector, keep your head perfectly still and do the following repeatedly:
- Move the mouse cursor to a random point in the window.
- Gaze at the mouse cursor
- Hold your gaze steady for 2 seconds to ensure it detects your pupils
- Click the mouse

When enough training data has been collected to yield a good fit (typically 10 to 30 clicks), a blue blur will appear centered at the predicted eye gaze position. You can do more clicks to further improve the fit. For best results, include a lot of points on the extreme left and right sides of the window. Be sure to keep your head as stationary as possible the whole time. (Don't move OR rotate your head at all.)

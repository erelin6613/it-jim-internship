### Task 1

The solution is not perfect and I bet there is much better one out there.

## What worked?

* converion to HSV colorspace
* thresholding
* decomposition to color channels (rgb), tresholding each of them and summation of results

## What did not work?

* opening/closing/operations with erosion and dilation in general
* Histogram eqalization, CLAHE in case of color decomposition
* Tresholding on the channels of LAB, LUV colorspaces (at least not as good as expected)

## Where were problems?

* adaptive thresholding raised an assertion error in the case with HSV colorspace
```cv2.error: OpenCV(4.2.0) /io/opencv/modules/imgproc/src/thresh.cpp:1647: error: (-215:Assertion failed) src.type() == CV_8UC1 in function 'adaptiveThreshold'```

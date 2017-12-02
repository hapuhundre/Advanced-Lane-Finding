## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

车道线检测，与最初的车道线检测项目相比，该项目有以下改进：

* 修正图像变形(distortion)
* 应用透视变换，将车道线的图像由斜视图变换为俯视图
* 结合色彩空间、sobel算子等更稳定地提取出车道线特征
* 应用多项式插值，适应车道线曲率变化较大的情况
* 标记出安全的行车空间

### 项目简介

项目实施过程如下:

- [x] 标定相机内参并存储其参数

      ----------------------------------image process pipeline----------------------------------------------

- [x] 将图像进行扭曲修正

- [x] 选取 颜色/梯度 阈值

- [x] 透视变换

      -----------------------------------*---------------------------------------------------------------------------

- [x] Detect lane lines

- [x] Determine the lane curvature


和proj1相比，实现步骤的区别在于图像处理的流程和特征点插值方法的改进。


The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!




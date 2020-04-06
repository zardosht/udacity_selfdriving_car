## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output2.png)

In this project we detect the lane lines under more challenging real conditions, including lane cruvatures, change of lighting and lane colors, shadows, and different road conditions. 

The **input** is camera videos of a vechicle driving on a highway and the **output** is the video annotated with and overlay that shows the detected lane, as well as detected readius of curvatures and the vehicle offset from center of the lane. 

Thre [writeup](https://github.com/zardosht/udacity_selfdriving_car/blob/master/P2_Advanced_Lane_Finding/writeup_template.md) explains steps of the pipeline in details. 

The requirements to pass the porject are listed [here](https://review.udacity.com/#!/rubrics/1966/view)


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

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are result of testing the pipeline on single frames.  The resulting imaages from each stage of the pipeline are stored in the folder called `output_images`. 

The pipeline is then applied to the `project_video.mp4` and the output is `project_video_output.mp4` 


I will also test the pipeline on `challenge_video.mp4` and the `harder_challenge.mp4` video. 



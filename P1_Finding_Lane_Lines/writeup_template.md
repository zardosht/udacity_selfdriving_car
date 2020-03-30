# **Finding Lane Lines on the Road** 

## Writeup



---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
 
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline does the following steps in order on the input image. The output is the input image with solid lane lines drawn on it. 

I did not modify the draw_lines() function; becuase it must do only what the name says: drawing lines. Instead I added two other helper functions: one for grouping the line segments of the Hough lines into left and right lanes, and one for estimating a solid line for the left and right lane from corresponding groups of line segments. 

**Pipeline Steps**: 

1. **Grays and Blur:** Convert the image to grayscale and apply a Gaussian blur with `kernel_size = 7`. I tried different values of the kernel, and 7 provided best results. 
2. **Canny edge detection:** I tried different values for thresholds, and given the good contrast and lighting conditions in the image a high value for both thresholds gave best results. 
3. **RoI mask:** After detecting the edges, I applied a RoI mask. The RoI has the shape of a trapezoid. The bottom side is the line between the bottom corners of the image. The top side is at `y = 0.6 * image_height`. I chose 0.6 by testing different values and checking the results. The x-coordinates of the top corners are hardcoded. 
4. **Hough lines:** I then compute the Hough lines on the masked edges image. The parameters for the Hough lines calculation were chosen experimentally to find best resutls on the test images. I chose a small value for `min_line_length = 5` so that the small lane line segments in the far distance are considered. 
5. **Group line segments:** The resulting line segments were grouped based on their slope into three groups: left lane, right lane, and discard. Assuming that the lines with a sharp slope belong to either lane lines, I defined a threshold for the slope. Any line withe the absolute value of slope below the threshold were discarded. These are normally vertical lines from the cars in the scene. The remaining lines were grouped based on the sign of their slope: positive for left lane, negative for right lane. 
6. **Estimate lane lines:** From each group of line segments, I estimated a solid line for the lane. The lane line is the best fitted line that passes through middle points of all line segments in a group.
7. **Draw lane lines:** Finally, the resulting lane lines were drawn on the input image. 



### 2. Identify potential shortcomings with your current pipeline

* Many values are hard-coded (e.g. the kernel for Gaussian blur, and all other parameters). These values are tuned for the input data, which consists of images with very good lighting condition and high contrast between lane markings and the road. The algorithm is not robust and does not return good results as soon as lighting condition or the contrast between road markings and the asphalt changes. For example see the results on the seconds 4 to 6 of the `challenge.mp4` video. 

* This approach only works for straight roads, and does not work well for curved lane lines. 

* The RoI is based on the assumption that road is not bumpy, and the camera is placed at the middle of the car. Also the top side of the RoI Trapezoid. 

* I also don't use color thresholding. This could possibly improve the results and reduce the required computations

* Also, fitting the lane line to middle point of line segments is potentially costly to compute. 


### 3. Suggest possible improvements to your pipeline

* Using adaptive thresholding and other ways for dynamically determining good parameters for the Canny and Hough algorithms. 

* A more efficient way for fitting the lane line. 

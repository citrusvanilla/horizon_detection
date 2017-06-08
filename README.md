# Vision-Based Maritime Horizon Detection with Dual Global and Local Objectives

The following notebook walks the user through an implementation of a computer vision-based horizon detection that uses a two-stage objective that greatly reduces computational overhead compared to an exhaustive search approach.

The first objective ("global") attempts to find a narrow range of combinations of "pitch" and "roll" (attitude and angle) corresponding to a halfplane that likely subdivdes the sky from the rest of the image.  The second objective ("local") searches exhaustively through these combinations to find the halfplane that maximizes the difference in average intensity of the two halfplanes in the immediate viscinity of the halfplane.

Compared with an exhaustive search of pitch and roll combinations performed at the outset, this method obtains perfect accuracy on our datset with a full order-of-magnitute less computations.  This method benefits from the assumption that a "sky" as represented by image data has higher intensity values than the ground pixels (higher mean), and that the sky has higher consistency of representation (lower variance).  A "coefficient of variance" calculation (ratio of mean to variance) performed on the full range of angle/attitude for all images in the set suggests that--at least for our data--the optimization surface is approximately convex, allowing for confident use of subsampling in the global objective.

Exhaustive search over the second objective for our dataset in the data exploration phase however reveals numerous local maxima- therefore we must employ exhaustive search over the subregion identified in the global objective in runtime.  Regardless, we observe no loss of accuracy on out output.

The method is demonstrated on a selection of maritime images whose time-of-day, glare effects, and ground-truth horizon location within the frame and varied.

### Software and Library Requirements
* Python 2.7.11
* Jupyter Notebook 4.2.2
* Numpy 1.11.2
* matplotlib 1.5.2
* OpenCV 3.2.0

## Goals
This repository demonstrates a novel solution to horizon detection for video streams of maritime images that reduces computational requirements as compared with common exhaustive approaches, such as that outlined by [Ettinger et al](https://www.researchgate.net/profile/Martin_Waszak/publication/2494734_Towards_Flight_Autonomy_Vision-Based_Horizon_Detection_for_Micro_Air_Vehicles/links/5441579b0cf2a76a3cc7de60.pdf).

## Key Processes
1. Solve global objective by optimizing attitute/angle objective surface.
2. Solve local objective by exhaustive search over range defined in (1).

## Code Organization

File | Purpose
------------ | -------------
two_objectives_horizon_detection.ipynb |	iPython Notebook for user-friendly implementation and data exploration.
two_objectives_horizon_detection.py | Python file containing main, preprocessing steps and subroutines.

## Getting up and running

While in the `two_objectives_horizon_detection` directory, enter the following in the command line:

> ipython notebook two_objectives_horizon_detection.ipynb

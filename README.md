### Demo

```scan.py matches.avi```

and wait until a pointcloud pops up! :D

Enjoy

### Short project description:

A DYI laser scanner based on Rpi, laser pointer, camera and rotating plate. 
The main script ```scan.py``` takes a video as an input and generates a pointcloud.

The calibration stages:
Intrinsic camera calibration -> laser plane calibration with respect to the camera -> plate rotation axis calibration with respect to the camera. 
All the calibration stages use charuco board as the reference. 

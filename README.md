# hyperspectral-imaging-preprocessing
This repository contains scripts for transmittance/reflectance calibration (same formula) using an reference images and for image mosaicing/stitching. It also contains various scripts for plotting hyperspectral data.

To run the scripts you need the packages: Numpy, OpenCV, SciPy, Pillow and SpectralPython. 

The full workflow from radiance images to a mosaiced image is: 1. Make a clean white reference without lines containing dust or debris (clean_white_reference_interactive.py) 2. Calibrate files using the cleaned white reference (calibrate_files.py) 3. Create a mosaic of two hyperspectral images using (mosaic_images.py). 

In addition, 3D models for making a hyperspectral transmission imaging setup suitable for any rectangular microscope slide. The .f3d files are easiest to customize, but require Autodesk Fusion 360. The .step files are customizable and the .stl files are most suitable for directly importing into a slicer for 3D-printing. 
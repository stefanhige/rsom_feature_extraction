**How to label epidermis**

Requirements
- python 3.8.5 (others might work too)
- imageio
- matplotlib
- scipy
- numpy
- nibabel
- scikit-image
- skimage

**Create MIP3D for easy labeling**

Starting from the MATLAB files, we create nii.gz files. RSOM data is of dimension 333 x 171 x depth.
In the direction of RSOM data that has 171 pixels, we perform MIPS of each 19 voxels. The resulting volume is 333 x 9 x depth.
Each of these 9 layers can be easily annotated using ITK-SNAP
In the process of calling this script, depth will be fixed to 500 (either extended or cropped).

`python prepare_for_labeling.py --mat-dir /path/to/matlab/data --output-dir /path/where/to/put/nii/files`

Open file *_mip3d.nii.gz in ITK_SNAP

1. On the left side, right click on the layer, got to multi-component-display and choose RGB.
2. Tools -> Reorient Image -> New Orientation -> ASR
3. Choose **Label 1** as Active Label for the segmentation
4. For each of the 9 MIPs, create polygon with **Label 1** and click "accept"
5. Be sure to not forget to label any of the 9 slices. After finishing, save the segmentation with the same name, except ending is *_mip3d_l.nii.gz. Can be saved in same directory.

**Interpolate annotations**
In order to retrieve the full label, the volume of 333 x 9 x 500 is going to get interpolated to 333 x 171 x 500.
Label directory and output directory may be the same.
`python interpolate_label.py --label-dir path/where/to/put/nii/files` --output-dir path/where/to/put/nii/files

Verify the interpolation worked by loading the respective *_rgb.nii.gz and *_l.nii.gz files in ITK-SNAP.




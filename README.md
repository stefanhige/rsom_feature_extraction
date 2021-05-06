
# Layer and Vessel Segmentation in RSOM Images

complete training code.
Due to deletion of some parts of this repository there may be inconsistencies and minor bugs.
This code will most likely not run "as it is".


## Preprocessing

Preprocessing steps to generate .nii.gz files from MATLAB measurement files can be found in folder `prep`.
`prep_{layerseg, layerseg_interp, vessel}.py` are executable scripts.

## Layer Segmentation

The pytorch dataset class is in `laynet/_dataset.py`, the network architecture in `laynet/_model.py` and
the class interface for performing a ML experiment (both training and prediction) is in `laynet/layer_net.py`.

To run the layer segmentation, use the script `run_laynet.py` at the top level of the repository.

## Vessel Segmentation

Similar naming convention as for the layer segmentation.
The class interface for performing a ML experiment (both training and prediction) is in `vesnet/vesnet.py`.

To run the vessel segmentation, use the script `run_vesnet.py` at the top level of the repository.

## Application

`pipeline.py` runs the whole process from preproccessing to layer and vessel segmentation and visualization. However, consider looking at https://github.com/stefanhige/pytorch-rsom-seg instead.




# MA IBBM


# current bugs / tasks

## VESNET

* major: implement grad cam for segmentation

* patching is causing different prediction intensities? why? groupnorm?

* clean up module structure, move reused things to utils

* pred_adj cannot handle background images, bc dice is zero, even tho this is correct


* choose automatic probability depending on maximum of dice overlay
**in progress** need verification sweep that minimize_scalar does actually find the minimum
 plot a plot prob against dice, to see if there's one or multiple minima
 furthermore: implement function doing that for a set of volumes, and they might have multiple global minima
 **done**
 -> choose final cl_score out_score calculation to be executed with new cutoff


* clean up lossfunctions in general. maybe move calc_metrics to utils?

* loading different models directly in prediction

* device=torch.device('cpu') might not work, as many times in subfunctions there is .cuda() used

* major: implement possibility to pass sparsly annotated data.
  - either in different file, or mark unannotated area with label "2"
  - in dataloader, detect, if a file has label "2" inside, if yes, split
    into "loss mask" and actual label
    acutal_label = label == 1
    mask = label != 2    (2 -> 0   , other -> 1) multiply with prediction/unreduced loss/unreduced loss
    ** might not be needed, as I just cut away vessels from background. So label is zero everywhere**

* **currently using bce again**
  dice. handling when dice score is ~1e-14, which is probably when mask is empty in that patch.
  do we even care to backprop? or set dice to a higher value.. complicated
   -> it might acutally help to add some fraction of background dice, as this one is
      close to 0.99 in that case..
      **as of now:** ??

* more data with "red" noise
  **done**

* function for hyperpar sweep **done sweepVesNET.py**

* check if values passed to DeepVesselNet are not 0, as Blue channel accidentally used
  **done**

* make nan from calc_metrics acceptable in eval()
  **fixed... nan are not considered now**

* make debug() work when called from outside
  **fixed with workaround declare global debug inside __init__**

* remove addDuplicateDim
  **done**

* make prediction probability avaiable for global setting
  **fixed**

##  LAYERUNET

* adjust cutoff in z direction

* merge back coding style improvements from vesnet to layerunet 

* fill holes in prediction in xy direction?

* check if intensity augmentation helps for generalization
##  OTHERS





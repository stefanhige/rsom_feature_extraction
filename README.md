# MA IBBM


# current bugs / tasks

## VESNET

* on release branch / github version: pep8 mistakes

* major: huge cleanup for pipeline and stuff
         - checkout new branch of current status
         - continue cleanup on master
           - clean prep, remove unused features, create wrapper methods to do all at once
           - laynet is almost cleaned, but can remove debug output in predict function
           - vesnet change naming conventions, clean up metrics/dice mess,
             remove unused functions
           - clean up pipeline to be as short as possible, ready for release
           - for all classes and methods, write doc string!
           after cleanup:
           checkout release branch, and remove tons of unused scripts and utils from release branch
 
* major: needs a lot of testing: cut away reflection or emptieness in vessel preparation!
         -> this will break mip-label-overlay!
         postpone for future work, after finishing master thesis

* patching is causing different prediction intensities? why? groupnorm?
         **yes, it was the groupnorm, but we keep it anyways and try to
           use divs=(1,1,2)**

* device=torch.device('cpu') might not work, as many times in subfunctions there is .cuda() used

* clean up module structure, move reused things to utils

* pred_adj cannot handle background images, bc dice is zero, even tho this is correct

* major: check during eval/prediction mode if can get more memory,
         try setting requires_grad=False for input variable
         or override eval() method
   in progress: setting a flag in DeepVesselNet for memefficient forward pass, but is slower
   **finished**

* theres a bug in patch handling, for divs (1,1,1), must be fixed!!
   was this case not tested?
   **fixed, bug was in predict method**

* choose automatic probability depending on maximum of dice overlay
**in progress** need verification sweep that minimize_scalar does actually find the minimum
 plot a plot prob against dice, to see if there's one or multiple minima
 furthermore: implement function doing that for a set of volumes, and they might have multiple global minima
 **done**

* major: implement possibility to pass sparsly annotated data.
  - either in different file, or mark unannotated area with label "2"
  - in dataloader, detect, if a file has label "2" inside, if yes, split
    into "loss mask" and actual label
    acutal_label = label == 1
    mask = label != 2    (2 -> 0   , other -> 1) multiply with prediction/unreduced loss/unreduced loss
    **might not be needed, as I just cut away vessels from background. So label is zero everywhere**

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
* find out what test_loss.py does and include it in top level run_tests.py function

* there is .cuda() in _metrics.py implement optional parameter for torch.device

* merge back coding style improvements from vesnet to layerunet 
  **done, pretty much**  

* fill holes in prediction in xy direction?
  **not yet done, but probably too much work**

* check if intensity augmentation helps for generalization
  **too much work. drop that idea**

* adjust cutoff in z direction
  **done**

##  OTHERS





# MA IBBM


# current bugs / tasks

## VESNET

* major: implement possibility to pass sparsly annotated data.
  - either in different file, or mark unannotated area with label "2"
  - in dataloader, detect, if a file has label "2" inside, if yes, split
    into "loss mask" and actual label
    acutal_label = label == 1
    mask = label != 2    (2 -> 0   , other -> 1) multiply with prediction/unreduced loss/unreduced loss


* choose automatic probability depending on maximum of dice overlay

* clean up calc_metrics and move to utils?

* external dice function which can be used directly?

* loading different models directly in prediction

* device=torch.device('cpu') might not work, as many times in subfunctions there is .cuda() used

* check if values passed to DeepVesselNet are not 0, as Blue channel accidentally used

* make nan from calc_metrics acceptable in eval()
  **fixed... nan are not considered now**

* make debug() work when called from outside
  **fixed with workaround declare global debug inside __init__**

* remove addDuplicateDim
  **done**

* make prediction probability avaiable for global setting
  **fixed**

##  LAYERUNET
* fill holes in prediction in xy direction?

* check if intensity augmentation helps for generalization
##  OTHERS





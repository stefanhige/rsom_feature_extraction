# MA IBBM


# current bugs / tasks

## VESNET

* make nan from calc_metrics acceptable in eval()

* make debug() work when called from outside
  **fixed with workaround declare global debug inside __init__**

* loading different models directly in prediction

* remove addDuplicateDim

* make prediction probability avaiable for global setting
  **fixed**

* device=torch.device('cpu') might not work, as many times in subfunctions there is .cuda() used

* check if values passed to DeepVesselNet are not 0, as Blue channel accidentally used

##  LAYERUNET
* fill holes in prediction in xy direction?

* check if intensity augmentation helps for generalization
##  OTHERS





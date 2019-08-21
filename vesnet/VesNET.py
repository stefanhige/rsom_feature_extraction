# main class for VesNET
# Stefan Gerl
#
#
#



class VesNET():
    '''
    class for setting up, training of vessel segmentation with deep vessel net 3d on RSOM dataset
    Args:
        device              torch.device()              'cuda' 'cpu'



        to be determined
    '''
    def __init__(self,
                 device=torch.device('cuda')
                 ):
        self.model = Deep_Vessel_Net_FC( 

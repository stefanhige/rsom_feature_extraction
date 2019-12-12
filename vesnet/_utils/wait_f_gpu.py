import subprocess
import time
import torch
import os
import smtplib, ssl

# from VesNET import VesNET
# from lossfunctions import BCEWithLogitsLoss, dice_loss
# from deep_vessel_3d import DeepVesselNet


ctr = 0
print('waiting for gpu becoming free...')
while 1:
    
    cmd = 'nvidia-smi | grep -B 1 -m 1 10MiB | grep "|   [0-9]" -m 1 -o | grep [0-9] -o; exit 0'
    res = subprocess.check_output([cmd], shell=True)
    res = res.decode('utf-8').replace('\n','')
    if res is not '':
        break
    
    time.sleep(1)
    ctr += 1
    if not ctr % 60:
        print('Waiting', ctr/60, 'mins')

print('GPU Nr:', res)
print('train .....')



port = 465  # For SSL
password = b'\x66\x62\x73\x65\x72\x76\x65\x72\x77\x65\x72\x74'.decode('utf-8')
sender_email = 'fbserveribbm@gmail.com'
receiver_email = 'bluadzac@gmail.com'
message = """\
Subject: Free GPU

GPU {:s} is free.
""".format(res)




context = ssl.create_default_context()

with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)


# py_script = 'VesNET.py'

# subprocess.run(['tmux send -t \{next\} \"ipython\" ENTER'], shell=True)
# time.sleep(1)
# subprocess.run(['tmux send -t \{next\} \"run {:s}\" ENTER'.format(py_script)], shell=True)


# while 0:
#     DEBUG = None
#     DEBUG = True

#     root_dir = '/home/gerlstefan/data/vesnet/syntDataset/rsom_style_noisy'


#     desc = ('test train capability with deep deep ves net')
#     sdesc = 'idendity_from_scratch_hq0001_d6_bce'


#     model_dir = ''
            
#     os.environ["CUDA_VISIBLE_DEVICES"]=res
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#     train_dir = os.path.join(root_dir, 'train')
#     eval_dir = os.path.join(root_dir, 'eval')
#     out_dir = '/home/gerlstefan/data/vesnet/out'
#     pred_dir = '/home/gerlstefan/data/vesnet/annot_test_retrain_capability/eval'

#     dirs={'train': train_dir,
#           'eval': eval_dir, 
#           'model': model_dir, 
#           'pred': pred_dir,
#           'out': out_dir}


#     model_d6 = DeepVesselNet(channels = [2, 5, 10, 20, 40, 80, 1], 
#                              kernels=[3, 5, 5, 3, 3, 1],
#                              depth=6)
#     print(model_d6.count_parameters())

#     net1 = VesNET(device=device,
#                   desc=desc,
#                   sdesc=sdesc,
#                   dirs=dirs,
#                   divs=(2,2,2),
#                   offset=(7,7,7),
#                   model=model_d6,
#                   batch_size=1,
#                   optimizer='Adam',
#                   class_weight=10,
#                   initial_lr=1e-4,
#                   lossfn=BCEWithLogitsLoss,
#                   epochs=100,
#                   ves_probability=0.95,
#                   _DEBUG=DEBUG
#                   )

#     # CURRENT STATE
#     net1.train_all_epochs()








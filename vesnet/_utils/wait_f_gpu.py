import subprocess
import time
import torch
import os
import smtplib, ssl


ctr = 0
print('waiting for gpu becoming free...')
while 1:
    # old CUDA
    # cmd = 'nvidia-smi | grep -B 1 -m 1 10MiB | grep "|   [0-9]" -m 1 -o | grep [0-9] -o; exit 0'
    cmd = 'nvidia-smi | grep -B 1 -m 1 " 0MiB" | grep "|   [0-9]" -m 1 -o | grep [0-9] -o; exit 0'
    res = subprocess.check_output([cmd], shell=True)
    res = res.decode('utf-8').replace('\n','')
    if res is not '':
        break
    
    time.sleep(1)
    ctr += 1
    if not ctr % 60:
        print('Waiting', ctr/60, 'mins')

print('GPU Nr:', res)

port = 465  # For SSL
password = b'\x66\x62\x73\x65\x72\x76\x65\x72\x77\x65\x72\x74'.decode('utf-8')
sender_email = 'fbserveribbm@gmail.com'
receiver_email = 'receiver@gmail.com'
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



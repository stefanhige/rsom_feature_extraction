import smtplib, ssl
import time
import string 
import random
import subprocess

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

if 1:
    for i in range(1):

        time.sleep(0.1)
        print(i)

        port = 465  # For SSL
        password = b'\x66\x62\x73\x65\x72\x76\x65\x72\x77\x65\x72\x74'.decode('utf-8')
        sender_email = 'fbserveribbm@gmail.com'
        receiver_email = 'a.g.l@gmx.net'
        message = """\
        die aufloesung {:d}

        Hallo Andreas G Lehner,

        {:s}

        Bitte leiten sie diese mail an 99 personen weiter,
        ansonsten wird ihre opa 99 jahre von boeseon emails
        heim ge sucht.


        Sie haben gewonnen {:d} Euros. viel spasss
        Bitte antworten Sie auf diese mail mit ihre bankverbindung und pin
        damit ueeberweisen moeglich ist


        random buchstaben um spamfilter auszutrichsen=?
        {:s}
        {:s}

        ##############
        Hier der code
        #############

        {:s}


        """.format(i, 
                randomString(77),
                random.randint(1000, 1000000), 
                randomString(25), 
                randomString(35),
                subprocess.check_output(['cat ./send_mail.py'], shell=True).decode('utf-8'))



        context = ssl.create_default_context()

        with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)


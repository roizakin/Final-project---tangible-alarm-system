
import time
import sounddevice as sd  # ספריית פייתון לניהול הקלטת קול
import numpy as np
from scipy.io.wavfile import write
import Led
from test import check
from Receiver import record_audio


fs = 44100  # קצב דגימה
duration = 10  # משך הקלטה בשניות
recording_interval = 10  # משך הזמן בין כל הקלטה


# לולאה אינסופית לביצוע הקלטה ועיבוד
while True:
    audio_data = record_audio(duration, fs)  # ביצוע הקלטה
    prediction = check(audio_data)   # עיבוד האודיו
    if prediction:
        Led.trigger_alert()
        prediction=0

   # print("Prediction:", prediction)
    time.sleep(recording_interval)  # המתנה לפני הקלטה הבאה


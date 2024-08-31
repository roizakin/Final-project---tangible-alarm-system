import time
import sounddevice as sd  # ספריית פייתון לניהול הקלטת קול
import numpy as np
from scipy.io.wavfile import write
import Led
#from my_machine_learning_algorithm import predict_audio  # האלגוריתם לפענוח של האודיו


# פונקציה לביצוע הקלטת קול
def record_audio(duration, fs):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording finished.")
    return recording.flatten()

# פונקציה להפעלת הקלטה מוקלטת
def play_recorded_audio(audio_data, fs):
    print("Playing recorded audio...")
    sd.play(audio_data, fs)
    sd.wait()
    print("Playback finished.")




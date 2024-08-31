import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
import os

# הגדרות
fs = 44100  # תדר דגימה
duration = 10  # משך זמן ההקלטה בשניות
output_folder = 'D/D(17)'  # תיקייה לשמירת ההקלטות
audio_file = r'C:\ALARM\A\0 (17).wav'  # שם קובץ האודיו שיש להפעיל בלופים

# יצירת התיקייה אם אינה קיימת
os.makedirs(output_folder, exist_ok=True)

# קריאת קובץ האודיו
audio_data, samplerate = sf.read(audio_file)

# מספר חזרות נדרש (כדי להבטיח מספיק זמן השמעה)

repeats = (30 * duration * fs) // len(audio_data) + 1
audio_data = np.tile(audio_data, (repeats, 1))


# פונקציה להפעלת האודיו והקלטה בו זמנית
def play_and_record(audio_data, samplerate, duration, output_folder):
    stream = sd.InputStream(samplerate=samplerate, channels=2)
    sd.play(audio_data, samplerate)

    with stream:
        for i in range(30):
            print(f'מתחיל להקליט {i}...')
            recording = stream.read(int(duration * samplerate))[0]
            file_name = os.path.join(output_folder, f'recording_{i}.wav')
            write(file_name, samplerate, recording)  # שמירת ההקלטה כקובץ WAV
            print(f'נשמר קובץ {file_name}')


# קריאה לפונקציה להפעלת האודיו והקלטה
play_and_record(audio_data, samplerate, duration, output_folder)

print('הקלטות הושלמו ונשמרו בהצלחה.')
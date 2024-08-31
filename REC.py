import sounddevice as sd
from scipy.io.wavfile import write
import os

# הגדרות
fs = 44100  # תדר דגימה
duration = 10  # משך זמן ההקלטה בשניות
output_folder = 'recordings'  # תיקייה לשמירת ההקלטות

# יצירת התיקייה אם אינה קיימת
os.makedirs(output_folder, exist_ok=True)

# הקלטה ושמירה של 20 קבצים
for i in range(20):
    print(f'מתחיל להקליט {i}...')
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # ממתין לסיום ההקלטה
    file_name = os.path.join(output_folder, f'recording_{i}.wav')
    write(file_name, fs, recording)  # שמירת ההקלטה כקובץ WAV
    print(f'נשמר קובץ {file_name}')

print('הקלטות הושלמו ונשמרו בהצלחה.')

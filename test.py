import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import librosa
import librosa.display
import sounddevice as sd
from scipy.io.wavfile import write

# Define the class mapping
class_mapping = {
    0: 'Emergency Sirens',
    1: 'Ambulance Sirens',
    2: 'Police Sirens',
    3: 'Y(1)',
    4: 'Y(2)',
    5: 'Y(3)',
    6: 'Y(4)',
    7: 'Y(5)',
    8: 'Y(6)',
    9: 'Y(7)',
    10: 'Y(8)',
    11: 'Y(9)',
    12: 'Y(10)',
    13: 'Y(11)',
    14: 'Y(12)',
    15: 'Y(13)',
    16: 'Y(14)',
    17: 'Y(15)',
    18: 'Y(16)',
    19: 'Y(17)',
}

# Define transformations for inference
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size used in training
    transforms.ToTensor(),  # Convert to tensor
])


# Function to predict the label of an image using the specified model
def predict_image(model, image_path, class_mapping):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = inference_transforms(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Set model to evaluation mode
    model.eval()

    # Predict the label
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    # Map index to class label
    predicted_class = class_mapping.get(predicted_label, "Unknown")

    return predicted_label, predicted_class, probabilities


# Load the saved PyTorch model 2
model2_path = 'C:\\Users\\roiza\\PycharmProjects\\Final_project\\mobilenet_v2_model.pth'
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model2 = mobilenet_v2(weights=weights)
model2.classifier[1] = nn.Linear(model2.last_channel, 20)
model2.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
model2.eval()

# Load the saved PyTorch model 1
model1_path = 'C:\\Users\\roiza\\PycharmProjects\\Final_project\\mobilenet_v2_weights.pth'
model1 = mobilenet_v2(pretrained=False)
model1.classifier[1] = nn.Linear(model1.last_channel, 2)
model1.load_state_dict(torch.load(model1_path, map_location=torch.device('cpu')))
model1.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2.to(device)
model1.to(device)

# Define the class mapping for model 1
class_mapping1 = {
    0: 'Alarm',
    1: 'No alaram'
}


# Function to create spectrogram from audio
def create_spectrogram(y, sr, image_file):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr, ax=ax)
    fig.savefig(image_file, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def process_audio(y, sr, output_file):
    # Create a single spectrogram from the entire audio file
    create_spectrogram(y, sr, output_file)


def create_pngs_from_wavs(input_file, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    y, sr = librosa.load(input_file)
    output_file = os.path.join(output_path, os.path.basename(input_file).replace('.wav', '.png'))
    process_audio(y, sr, output_file)


# Function to record audio
#def record_audio(duration, fs, audio_file, device_id):
    # print("Recording...")
    # recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, device=device_id)
    # sd.wait()  # Wait until recording is finished
    # write(audio_file, fs, recording)  # Save as WAV file
    # print("Recording saved to", audio_file)

def check(record):
        # Paths
        output_base_dir = 'C:\\Users\\roiza\\PycharmProjects\\Final_project\\PIC'
        output_spectrogram_dir = os.path.join(output_base_dir, 'Spectograms')
        audio_file = os.path.join(output_base_dir, 'recorded_audio.wav')

        # Ensure the output directory exists
        os.makedirs(output_spectrogram_dir, exist_ok=True)

        # Recording settings
        fs = 44100  # Sample rate
        duration = 10  # Duration of recording in seconds
        desired_microphone_id = 6  # Set the desired microphone ID (replace with your microphone's ID)

        # Record audio, create a single spectrogram, and predict
        #record_audio(duration, fs, audio_file, desired_microphone_id)

        write(audio_file, fs, record)  # Save as WAV file
        print("Recording saved to", audio_file)

        create_pngs_from_wavs(audio_file, output_spectrogram_dir)  # Create spectrograms from the audio file

        # Predict the label of the generated spectrogram
        spectrogram_files = [f for f in os.listdir(output_spectrogram_dir) if f.endswith('.png')]
        if spectrogram_files:
            image_path = os.path.join(output_spectrogram_dir, spectrogram_files[0])

            # First pass through Model 1
            predicted_label1, predicted_class1, probabilities1 = predict_image(model1, image_path, class_mapping1)
            print(
                f"Model 1 - The predicted label for the image '{spectrogram_files[0]}' is: {predicted_label1} ({predicted_class1})")
            print(f"Model 1 - Probabilities: {probabilities1.cpu().numpy()}")

            # If Model 1 predicts 'Siren classes', pass through Model 2
            if predicted_label1 == 0:
                predicted_label2, predicted_class2, probabilities2 = predict_image(model2, image_path, class_mapping)
                print(
                    f"Model 2 - The predicted label for the image '{spectrogram_files[0]}' is: {predicted_label2} ({predicted_class2})")
                print(f"Model 2 - Probabilities: {probabilities2.cpu().numpy()}")
                return predicted_label2 == 2
            else:
                print("Model 2 was not used as Model 1 did not classify the image as a Siren class.")
                return 0
        else:
            print("No spectrogram was generated.")
            return 0
import pyaudio
import wave
import numpy as np
import torch
from torch.nn.functional import pad
from torchaudio.transforms import MelSpectrogram
from models.model import CNN_min

NUM_SAMPLES = 100000
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 2 
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1
CHUNK = 1024
RECORD_SECONDS = 5
SAMPLES = int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)
POSITIONS = [180, 135, 90, 45, 0]
MODEL_PATH = "models/cnn-min-melmodel.pth"

p = pyaudio.PyAudio()
mel = MelSpectrogram(
        sample_rate = RESPEAKER_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64)

model = CNN_min(W = 64, H = 196)
state_dict = torch.load(
        MODEL_PATH, 
        weights_only=False, 
        map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
torch.set_printoptions(precision=2)

model.eval()
while(True):
    stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)
    
    print("INICIO GRABACION")
    
    frames = [] 
    
    for i in range(0, SAMPLES):
        data = stream.read(CHUNK)
        np_data = np.frombuffer(data, dtype=np.int16)
        np_data = np.stack((np_data[::2], np_data[1::2]), axis=0)
        frames.append(np_data)

    frames = np.concatenate(frames, axis=1)
    audio = torch.Tensor(frames)

    if audio.shape[1] < NUM_SAMPLES:
        n_samples = NUM_SAMPLES - audio.shape[1]
        audio = pad(audio, (0, n_samples))

    mels = torch.stack((mel(audio[0]), mel(audio[1])), dim=0)

    with torch.no_grad():
        pred = model(mels.unsqueeze(0))

    _, index = torch.max(pred,1)
   
    print("Probabilidades =", torch.nn.functional.softmax(pred, dim=1) * 100)
    print("Orientacion estimada: ", POSITIONS[index])
    print("FIN GRABACION")
    
    stream.stop_stream()
    stream.close()
p.terminate()


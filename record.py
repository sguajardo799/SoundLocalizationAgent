import pyaudio
import wave
import pandas as pd
from time import sleep

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 2 
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1  
CHUNK = 1024
RECORD_SECONDS = 7 


FILES_CREATED = 50
ORIENTATION = 0 
DATA_PATH = "collected_data/" 
ROOT_FILENAME= "departamento.wav"


p = pyaudio.PyAudio()
for i in range(FILES_CREATED):
    #p = pyaudio.PyAudio()
    
    stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)
   
    
    print("======================Configuración Lista======================")

    print("* recording")
    frames = []

    for j in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* done recording")

    stream.stop_stream()
    stream.close()
    ##p.terminate()
   
    OUTPUT_FILENAME = DATA_PATH + str(i) + "_" + str(ORIENTATION) + ROOT_FILENAME

    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(RESPEAKER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


p.terminate()

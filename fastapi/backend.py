from fastapi import FastAPI, File
import numpy as np
import torch
import os
import base64
import cv2
app = FastAPI()
import pickle
from models.model import DecoderRNN, EncoderCNN

embed_size = 256#<-
hidden_size = 512#<-
vocab_size = 9955
encoder_file = 'v3encoder-4.pkl'
decoder_file = 'v3decoder-4.pkl'
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('', decoder_file)))
# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)

dicctionary= pickle.load( open( "dicctionary.pkl", "rb" ) )
def clean_sentence(output,dicttionary):
    sentence=[]
    for i in output:
        sentence.append(dicctionary[i])
    indices = [i for i, s in enumerate(sentence) if '<end>' in s]
    sentence=sentence[1:indices[0]]
    sentence=' '.join(sentence)
    return sentence

@app.post("/predict")
async def analyse(image_file_read: bytes = File(...)):
    file = base64.b64encode(image_file_read)
    jpg_original = base64.b64decode(file)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    original_image = cv2.imdecode(jpg_as_np, flags=1)

    #return original_image.shape

    im = original_image.copy()
    im = im / 255
    im = torch.tensor(im.transpose(2, 0, 1), dtype=torch.float32)
    encoder.eval()
    with torch.no_grad():
        image = im.to(device)
        # Obtain the embedded image features.
        features = encoder(image.unsqueeze(0)).unsqueeze(1)
        # Pass the embedded image features through the model to get a predicted caption.
    output = decoder.sample(features)
    sentence = clean_sentence(output, dicctionary)
    return sentence

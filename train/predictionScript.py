import cv2
import numpy as np
import os
import torch
from model import EncoderCNN, DecoderRNN
import pickle

import matplotlib.pyplot as plt


print("predict script v1")
embed_size = 256#<-
hidden_size = 512#<-
vocab_size = 11543
encoder_file = 'saved_models/v3encoder_7.pkl' 
decoder_file = 'saved_models/v3decoder_7.pkl'

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
print(encoder.to(device))
decoder.to(device)
print(decoder.to(device))

dicctionary= pickle.load( open( "dicctionary.pkl", "rb" ) )

def clean_sentence(output,dicttionary):
    sentence=[]
    for i in output:
        sentence.append(dicctionary[i])
    indices = [i for i, s in enumerate(sentence) if '<end>' in s]
    sentence=sentence[1:indices[0]]
    sentence=' '.join(sentence)
    return sentence


def predictImage(path_to_image):
    imgoriginal=cv2.imread(path_to_image)
    im=imgoriginal.copy()
    sentence=describeImage(im)
    print('example sentence:', sentence)
    plt.imshow(cv2.cvtColor(imgoriginal, cv2.COLOR_BGR2RGB))
    plt.show()
    
def describeImage(im):
    im=im/255
    im=torch.tensor(im.transpose(2, 0, 1),dtype=torch.float32)
    encoder.eval()
    with torch.no_grad():
        image = im.to(device)
        # Obtain the embedded image features.
        features = encoder(image.unsqueeze(0)).unsqueeze(1)
        # Pass the embedded image features through the model to get a predicted caption.
    output = decoder.sample(features)
    sentence = clean_sentence(output,dicctionary)
    return sentence
import torch
import torch.nn as nn
import torchvision.models as models

print("version 2")
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size= vocab_size
        self.num_layers=num_layers
        
        # Embedding layer
        #https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
        #https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)#
        
        # The LSTM 
        self.lstm = nn.LSTM(input_size=self.embed_size, \
                            hidden_size=self.hidden_size, # LSTM hidden units 
                            num_layers=self.num_layers, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )
        
        # The linear layer 
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)         
        
    def forward(self, features, captions):
        #Embedd
        embeddings = self.word_embeddings(captions[:, :-1]  )
       
        # Stack the features and captions
        embed = torch.cat((features.unsqueeze(1), embeddings), dim=1) 
        
        #LSTM
        lstm_out, self.hidden = self.lstm(embed) 
        #lstm_out, _ = self.lstm(embeddings.view(len(captions), 1, -1))
        
        # Fully connected layer
        tag_space = self.linear(lstm_out)
        #tag_space = self.linear(lstm_out.view(len(sentence), -1))
        return tag_space
        

        
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)         # hiddens: (1, 1, hidden_size)
            outputs = self.linear(lstm_out.squeeze(1))       # outputs: (1, vocab_size)
            _, predicted = outputs.max(dim=1)                    # predicted: (1, 1)
            res.append(predicted.item())
            
            inputs = self.word_embeddings(predicted)             # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (1, 1, embed_size)
        return res
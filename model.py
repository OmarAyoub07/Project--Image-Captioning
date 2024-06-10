import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        self.resnet_model = models.resnet18(pretrained=True)
        for param in self.resnet_model.parameters():
            param.requires_grad_(False)
            
        # # Unfreeze layer4
        # for param in self.resnet_model.layer4.parameters():
        #     param.requires_grad = True

        # # Unfreeze the fully connected layer (fc)
        # for param in self.resnet_model.fc.parameters():
        #     param.requires_grad = True
        
        modules = list(self.resnet_model.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(self.resnet_model.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # TODO: Complete this function
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        # TODO: Complete this function
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = (self.lstm(embeddings))
        outputs = (self.fc1(hiddens))
        return outputs

    # TODO: Enhance the method to get sample 
    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        
                    
        predicted_sentence = []
        
        for i in range(max_len):
            
            # Run the LSTM
            hiddens, states = self.lstm(inputs, states)
            
            # Get the output scores
            outputs = self.fc1(hiddens.squeeze(1))
            
            # Get the predicted word index
            _, predicted = outputs.max(1)
            
            # Append the predicted word index to the sentence
            predicted_sentence.append(predicted.item())
            
            # Prepare the next input
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence

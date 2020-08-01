
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

###################################
# Pytorch simple Seq-Seq (char-char) model
# 
# 
# Model looks like
#   c_s(1)  c_s(2)  c_s(3)  c_s(3) ....
#    |        |       |       | 
#   LSTM    LSTM    LSTM    LSTM   ....
#    |        |       |       |    ....
#   FNN      FNN     FNN     FNN   ....
#    |        |       |       |   
#  c_t(1)   c_t(2)  c_t(3)  c_t(3) ....
#
###################################


#Train and infer on GPU if possible
train_on_gpu = torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


class AnnaKareninaTextData():

    def __init__(self):
        
        with open('data.txt', 'r') as f:
            self.text = f.read()
        self.chars = tuple(set(self.text ))
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.data = np.array([self.char2int[ch] for ch in self.text])

    def one_hot_encode(self, arr, n_labels):
        
        one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        one_hot = one_hot.reshape((*arr.shape, n_labels))
        return one_hot

    def get_batches(self, arr, batch_size, seq_length):

        batch_size_total = batch_size * seq_length
        n_batches = len(arr)//batch_size_total
        arr = arr[:n_batches * batch_size_total]
        arr = arr.reshape((batch_size, -1))

        for n in range(0, arr.shape[1], seq_length):
            x = arr[:, n:n+seq_length]
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y


class CharLevelRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(len(tokens), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(tokens))
      
    
    def forward(self, x, hidden):
                
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden


def train(net, dataAPI, epochs=10, batch_size=10, seq_length=50, 
         lr=0.001, clip=5, val_frac=0.1, print_every=10):

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(dataAPI.data)*(1-val_frac))
    data, val_data = dataAPI.data[:val_idx], dataAPI.data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in dataAPI.get_batches(data, batch_size, seq_length):
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            x = dataAPI.one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            net.zero_grad()
            output, h = net(inputs, h)
            
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in dataAPI.get_batches(val_data, batch_size, seq_length):

                    x = dataAPI.one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train() 
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        

        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):
    
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)



if __name__ == '__main__':

    dataAPI = AnnaKareninaTextData()    
    net = CharLevelRNN(dataAPI.chars, n_hidden=64, n_layers=1)
    print(net)

    train(net, dataAPI, epochs=1, batch_size=8, 
         seq_length=32, lr=0.001, print_every=10)

    print(sample(net, 100, prime='Anna', top_k=5))
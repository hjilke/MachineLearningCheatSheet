import torch.nn as nn
import torch.nn.functional as F

# A Simple CNN for classification
class ConvNet(nn.Module):

    def __init__(self, n_classes, n_hidden_units_fc):
        super(ConvNet, self).__init__()

        self.l1 = nn.Conv2d(3, 16, 3, padding=1)
        self.l2 = nn.Conv2d(16, 32, 3, padding=1)
        self.l3 = nn.Conv2d(32, 64, 3, padding=1)
        self.l4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d((2,2), (2,2))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(n_hidden_units_fc, n_classes)
        
    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = self.pool(x)
        x = F.relu(self.l2(x))
        x = self.pool(x)
        x = F.relu(self.l3(x))
        x = self.pool(x)
        x = F.relu(self.l4(x))
        x = self.pool(x)
        x = x.view(-1, n_hidden_units_fc)
        x = self.dropout(x)
        x = F.softmax(self.linear(x))

        return x


from torch import nn

"""
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one 

    def forward(self, x):
        return self.classifier(x), {}
"""

class Classifier(nn.Module):
    def _init_(self, num_classes):
        super(Classifier, self)._init_()
        self.linear = nn.Linear(1024, 8)
        self.batch_norm = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
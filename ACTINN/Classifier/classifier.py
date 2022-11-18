import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, output_dim=None, input_size=None):
        """

        The Classifer class: We are developing a model similar to ACTINN for good accuracy

        """
        if output_dim == None or input_size == None:
            raise ValueError(
                'Must explicitly declare input dim (num features) and output dim (number of classes)'
            )

        super(Classifier, self).__init__()
        self.inp_dim = input_size
        self.out_dim = output_dim

        # feed forward layers
        self.classifier_sequential = nn.Sequential(
            nn.Linear(self.inp_dim, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 25), nn.ReLU(),
            nn.Linear(25, output_dim)
            # SoftMax not needed for CrossEntropyLoss in PyTorch
            ## i.e. no need for nn.Softmax(dim=1)
        )
        self.labels = None
        self.sum_index = None
        self.train_index = None

    def set_meta(self, idx2, label_to_type_dict):
        self.sum_index, self.train_index = idx2
        self.labels = label_to_type_dict

    def forward(self, x):
        """

        Forward pass of the classifier

        """

        out = self.classifier_sequential(x)

        return out

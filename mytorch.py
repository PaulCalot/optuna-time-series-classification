import torch
import numpy as np

class TorchClassifier:
    def __init__(self, model, lr, epochs, batch_size) -> None:
        self.model = model 
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def fit(self, x, y):
        train(self.model, x, y, self.lr, self.epochs, self.batch_size, self.device)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.Tensor(x).to(self.device)
            pred = self.model(x_tensor)
        self.model.train()
        return pred.cpu().numpy()

def train(model, x, y, lr, epochs, batch_size, device):
    # we suppose GPU is powerful enough to handle full dataset, if available
    dataset = ClassificationDataset(torch.Tensor(x).to(device), torch.LongTensor(y).to(device))
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, lengths)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_list = []
    # valid_loss_list = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(epoch, model, optimiser, loss_function, train_loader)
        # valid_loss = evaluate(valid_loader, model, loss_function)

        train_loss_list.append(train_loss)
        # valid_loss_list.append(valid_loss)
    return train_loss_list # , valid_loss_list

def evaluate(dataloader, model, loss_function):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            loss = loss_function(pred, y)
            loss_list.append(loss.cpu().numpy())
    model.train()
    return np.mean(loss_list)

def train_one_epoch(epoch_index, model, optimiser, loss_function, data_loader):
    loss_list = []
    for x, y in data_loader:
        pred = model(x)
        optimiser.zero_grad()
        loss = loss_function(pred, y)
        loss.backward()
        optimiser.step()
        loss_list.append(loss.detach().cpu().numpy())
    return np.mean(loss_list)


class TimeSeriesClassificationNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation_fn, number_of_classes) -> None:
        super().__init__()
        assert(activation_fn in ['softmax', 'logsoftmax'])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=0)
        self.output_layer = torch.nn.Linear(in_features=hidden_size,
                                            out_features=number_of_classes,
                                            bias=True)
        if(activation_fn == 'softmax'):
            self.activation_fn = torch.nn.Softmax(dim=1)
        # default
        self.activation_fn = torch.nn.LogSoftmax(dim=1)

    def forward(self, X):
        if(len(X.shape) == 2):
            X = torch.unsqueeze(X, dim=-1)
        X, h = self.rnn(X) # size is batch size x sequence length x output_size = input_size
        X = X[:, -1, :] # selecting last output
        X = self.output_layer(X)
        X = self.activation_fn(X)
        return X

    def init_weights(self):
        pass


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    

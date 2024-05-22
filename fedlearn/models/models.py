import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class fedmodel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_params(self, flat_params):
        prev_ind = 0
        for param in self.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:(prev_ind + flat_size)].view(param.size())
            )
            prev_ind += flat_size
    
    def get_flat_params(self):
        params = [param.data.view(-1) for param in self.parameters()]
        flat_params = torch.cat(params).clone().detach()
        return flat_params
    
    def get_flat_grads(self):
        try:
            grads = [param.grad.data.view(-1) for param in self.parameters()]
            flat_grads = torch.cat(grads).clone().detach()
        except AttributeError:
            return None
        return flat_grads

class Logistic(fedmodel):
    def __init__(self, input_shape, output_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(input_shape, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 1:
            x = x.view([1, x.shape[0]])
        # print(np.prod(x.shape[1:]))
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.layer(x)
        x = self.sigmoid(x)
        return x
    
class OneHiddenLayerMLP(fedmodel):
    def __init__(self, input_shape, output_dim, hidden_dim=64):
        super(OneHiddenLayerMLP, self).__init__()
        self.layer_input = nn.Linear(input_shape, hidden_dim)
        self.layer_hidden = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.layer_input(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_hidden(x)
        return self.sigmoid(x)

class TwoHiddenLayerMLP(fedmodel):
    def __init__(self, input_shape, output_dim, hidden_dim=128):
        super(TwoHiddenLayerMLP, self).__init__()
        self.layer_input = nn.Linear(input_shape, hidden_dim)
        self.layer_hidden_1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_hidden_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.layer_input(x)
        x = F.relu(x)
        x = self.layer_hidden_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden_2(x)
        return self.sigmoid(x)


class CNNMnist(fedmodel):
    def __init__(self, num_class):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_class)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(fedmodel):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(fedmodel):
    def __init__(self, num_class):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet18(fedmodel):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        
        self.b1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, num_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x =  self.net(x)
        return self.sigmoid(x)
    

def ModelMapping(options):
    dataset = str(options['data']).lower()
    model = str(options['model']).lower()
    if dataset == 'mnist':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape': 784, 'num_class': 10}
        else:
            return {'input_shape': (1, 28, 28), 'num_class': 10}
    elif dataset == 'emnist':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape': 784, 'num_class': 62}
        else:
            return {'input_shape': (1, 28, 28), 'num_class': 62}
    elif dataset == 'fashionmnist':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape': 784, 'num_class': 10}
        elif model in ['resnet', 'resnet2']:
            return {'input_shape': (224, 224, 1), 'num_class': 10}
    elif dataset == 'synthetic':
        return {'input_shape': 60, 'num_class': 10}
    elif dataset == 'cifar':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape': 3 * 32 * 32, 'num_class': 10}
        else:
            return {'input_shape': (3, 32, 32), 'num_class': 10}
    elif dataset == 'celeba':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
        if model in ['resnet']:
            return {'input_shape': (3,128,128), 'num_class': 1}
    elif dataset == 'adult':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape': np.prod(options['num_shape']), 'num_class': 1}
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'compas':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape':np.prod(options['num_shape']), 'num_class': 1}
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'compas_1':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape':np.prod(options['num_shape']), 'num_class': 1}
        else:
            raise ValueError('{} doesnot support model {}!'.format(dataset, model))
    elif dataset == 'bank':
        if model in ['logistic', '2nn', '1nn']:
            return {'input_shape':np.prod(options['num_shape']), 'num_class': 1}
    else:
        raise ValueError('Not support dataset {}!'.format(dataset))

def choose_model(options):
    model_name = str(options['model']).lower()
    modelconfig = ModelMapping(options)
    options.update(modelconfig)
    modelconfig['output_dim'] = modelconfig.pop('num_class')
    if model_name == 'logistic':
        return Logistic(**modelconfig)
    elif model_name == '2nn':
        return TwoHiddenLayerMLP(**modelconfig)
    elif model_name == '1nn':
        return OneHiddenLayerMLP(**modelconfig)
    elif model_name == 'cnncifar':
        return CNNCifar(10)
    elif model_name == 'cnnmnist':
        return CNNMnist(10)
    elif model_name == 'resnet':
        return ResNet18(input_channels=3, num_channels=1)
    else:
        raise ValueError("Not support model: {}!".format(model_name))
    
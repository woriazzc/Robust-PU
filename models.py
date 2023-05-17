import torch.nn as nn


class normalNN(nn.Module):
    def __init__(self, x_dim):
        super(normalNN, self).__init__()
        self.fc1 = nn.Linear(x_dim, 100)
        self.fc2 = nn.Linear(100, 1)

        self.af = nn.ReLU(inplace=True)

        self.reset_para()

    def forward(self, x, return_fea=False):
        h = self.fc1(x)
        h = self.af(h)
        fea = h.detach().clone()
        h = self.fc2(h)
        if return_fea:
            return h.squeeze(), fea
        return h.squeeze()

    def reset_para(self):
        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)


class CNN(nn.Module):
    def __init__(self, x_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(96)
        self.b3 = nn.BatchNorm2d(96)
        self.b4 = nn.BatchNorm2d(192)
        self.b5 = nn.BatchNorm2d(192)
        self.b6 = nn.BatchNorm2d(192)
        self.b7 = nn.BatchNorm2d(192)
        self.b8 = nn.BatchNorm2d(192)
        self.b9 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

        self.af = nn.ReLU(inplace=True)

        self.reset_para()

    def forward(self, x, return_fea=False):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = h.reshape(h.shape[0], -1)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        fea = h.detach().clone()
        h = self.fc3(h)
        if return_fea:
            return h.squeeze(), fea
        return h.squeeze()

    def reset_para(self):
        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.kaiming_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

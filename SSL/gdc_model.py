from torch import nn

from torchvision import models

class ResnetGradCam(nn.Module):
    def __init__(self):
        super(ResnetGradCam, self).__init__()
        resnet152 = models.resnet152()

        # Remove linear layer
        modules = list(resnet152.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 3) # resnet

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        return x

# if __name__ == "__main__":
#     import torch
#     model = ResnetGradCam()
#     loaded = torch.load("../mulmodel_99.69_88.33_stanford.pth")
#     model.load_state_dict(loaded, strict=False)



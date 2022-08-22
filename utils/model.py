from abc import ABC
import torch
import torch.nn as nn
import torchvision.models as models
from utils.coatnet_pytorch import coatnet_0 , CoAtNet
import sys

class ResNet(nn.Module, ABC):
    def __init__(self, _model_):
        super(ResNet, self).__init__()
        self.model = _model_
        self.cnn = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


class Vgg16bn(nn.Module, ABC):
    def __init__(self, _model_):
        super(Vgg16bn, self).__init__()
        self.model = _model_
        self.cnn = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.fc = nn.Linear(25088, 3)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class ResNeXt(nn.Module, ABC):
    def __init__(self, _model_):
        super(ResNeXt, self).__init__()
        self.model = _model_
        self.cnn = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.fc = nn.Linear(in_features=2048, out_features=3)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class VisionTransformer(nn.Module,ABC):
    def __init__(self, _model_):
        super(VisionTransformer, self).__init__()
        self.model = _model_
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.encoder = self.feature_extractor[1]
        self.fc = nn.Linear(in_features=768, out_features=3)

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n,-1,-1)
        x = torch.cat([batch_class_token,x],dim=1)
        x = self.encoder(x)
        x = x[:,0]
        x = self.fc(x)
        return x




def baseline(name, n_blocks = [2,2,3,5,2], channels = [64,96,192,384,768],block_types = ['C','T','T','T'], pretrained=True):
    """
    :param name: name of the model
    :param pretrained: use pretrained model or not
    :return: model
    """
    try:
        if name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            return ResNet(model)
        elif name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            return ResNet(model)
        elif name == 'vgg16':
            model = models.vgg16_bn(pretrained=pretrained)
            return Vgg16bn(model)
        elif name == "resnext50":
            model = models.resnext50_32x4d(pretrained=pretrained)
            return ResNeXt(model)
        elif name == "resnext101":
            model = models.resnext101_32x8d(pretrained=pretrained)
            return ResNeXt(model)
        elif name == "ViT":
            model = models.vit_b_16(pretrained=pretrained)
            return VisionTransformer(model)
    
    except Exception as e:
        print(e)


if __name__ == "__main__":
    pass
    # m = baseline(name="baseline_resnext50")
    # print(list(m.cnn.children())[-3][-1])

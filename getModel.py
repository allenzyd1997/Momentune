import torch
import torch.nn as nn
from torchvision import models
from imageBaseModel import ImageClassificationBase

class ResNet50_C(ImageClassificationBase):
    def __init__(self, configs, namelist):
        super().__init__()
        print(configs)
        self.network = models.resnet50(pretrained=configs.pretrain)
        if configs.moco:
            checkpoint = torch.load(configs.moco_pth)
            checkpoint_dict = checkpoint['state_dict']
            for nt, k in enumerate(list(checkpoint_dict.keys())):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    checkpoint_dict[k[len("module.encoder_q") + 1:]] = checkpoint_dict[k]
                del checkpoint_dict[k]

            self.network.load_state_dict(checkpoint_dict,strict=False)

            print("ーーー　MoCo たいせっと装填完了　ーーー")
        else:
            print("Initialization finish")
        self.layernamelist = namelist
        self.network.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=configs.class_num)
        )

        self.aux_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        # reserve all the bn data
        self.bn_origin = {}
        for name, parameters in self.named_parameters():
            if 'bn' in name:
                self.bn_origin[name] = parameters


        self.freezeBN()


    def forward(self, x_val):
        return self.network(x_val)

    # def forward(self, x_val):
    #     # ABN
    #     x = self.network.conv1(x_val)
    #     a1 =  self.network.bn1(x)
    #     a2 =  self.aux_bn(x)
    #     x  =  self.network.relu(a1*0.5 + a2*0.5)
    #     # x = self.network.relu(a1)
    #     x  = self.network.maxpool(x)
    #     x  = self.network.layer1(x)
    #     x = self.network.layer2(x)
    #     x = self.network.layer3(x)
    #     x = self.network.layer4(x)
    #     x = self.network.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.network.fc(x)
    #     return x
    #


    def freeze(self, list):
        for name, parameters in self.named_parameters():
            if name in list:
                parameters.requires_grad = False

    def freezeBN(self):
        for name, parameters in self.named_parameters():
            if 'bn' in name:
                parameters.requires_grad = False

    def unfreezeBN(self):
        for name, parameters in self.named_parameters():
            if 'bn' in name:
                parameters.requires_grad = True

    def momenTuneBN(self, alpha = 0.3):
        checkpoint_dict = {}
        for name, parameters in self.named_parameters():
            if 'bn' in name:
                result = alpha * parameters + (1-alpha) * self.bn_origin[name]
            else:
                result = parameters
            checkpoint_dict[name[8:]] = result
        self.network.load_state_dict(checkpoint_dict, strict=False)
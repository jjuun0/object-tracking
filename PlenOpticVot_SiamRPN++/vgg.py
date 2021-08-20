import torch.nn as nn
import torch
import torchvision.models as models

origin_model = models.vgg16(pretrained=True)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # self.network = models.vgg16(pretrained=True)
        self.features = nn.Sequential(
            *list(origin_model.features.children())
        )

    def forward(self, x):
        # feature = self.network.features
        # out = feature(x)
        out = self.features(x)
        return out  # 7*7*512

model = VGG16()
def get_feature(x):
    network = model.cuda().eval()
    out = network(x)
    return out


if __name__ == '__main__':
    # # net = VGG16()
    # # print(net)
    test = torch.rand(1, 3, 255, 255)
    net = VGG16()
    print(net)

    # 파라미터 freeze
    for param in net.parameters():
        param.requires_grad_(False)

    out = net(test)
    print(out)

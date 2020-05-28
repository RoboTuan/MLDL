import torch.nn as nn
from torch.autograd import Function

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


''' 
Very easy template to start for developing your AlexNet with DANN 
Has not been tested, might contain incompatibilities with most recent versions of PyTorch (you should address this)
However, the logic is consistent
'''

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class RandomNetworkWithReverseGrad(nn.Module):
    # Changed **kwargs to num_classes, see if it's correct
    # kwargs = {'num_classes':7, 'num_domain_classes':2}
    def __init__(self, num_classes=1000):

        super(RandomNetworkWithReverseGrad, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), # input size is that of the previous output
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.dann_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), # input size is that of the previous output
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, alpha=None):
        features = self.features(x)
        #print(features.size())  # debug
        features = self.avgpool(features)
        #print(features.size())  # debug
         # Flatten the features:
        features = features.view(features.size(0), -1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(features, alpha)
            discriminator_output = self.dann_classifier(reverse_feature)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            # do something else
            class_outputs = self.classifier(features)
            return class_outputs



def alexnetDANN(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = RandomNetworkWithReverseGrad()

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    model.classifier[6] = nn.Linear(4096, kwargs['num_classes'])
    model.dann_classifier[6] = nn.Linear(4096, kwargs['num_domain_classes'])

    model.dann_classifier[0] = model.classifier[0]
    model.dann_classifier[1] = model.classifier[1]
    model.dann_classifier[2] = model.classifier[2]
    model.dann_classifier[3] = model.classifier[3]
    model.dann_classifier[4] = model.classifier[4]
    model.dann_classifier[5] = model.classifier[5]

    return model
import torch
import torchvision
import torch.nn as nn
import torchvision.models


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.layer_ = '''
         use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = [2, 7, 14]
        vgg19 = torchvision.models.vgg19(pretrained=True)
        # vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        # import torchvision.models as models
        #
        # # Use VGG19 model with default weights
        # model = models.vgg19(weights=models.vgg19.VGG19_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1] + 1])

    def forward(self, x):
        x = (x - 0.5) / 0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
        return features


def VGGPerceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss

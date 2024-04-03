import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitBlock(nn.Module):
    def __init__(self, split_ratio=(3/8)):
        super(SplitBlock, self).__init__()
        assert split_ratio <= 0.5
        self.split_ratio = split_ratio

    def forward(self, x):
        c = x.size(1)
        c1 = int(c * self.split_ratio)
        c2 = (c-c1)//2 
        out = torch.split(x, [c2,c2,c1], dim=1)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.375, fuse_ratio=0.5):

        super(BasicBlock, self).__init__()
        
        self.split = SplitBlock(split_ratio)
        cin = int(split_ratio*in_channels)
        cout = int(fuse_ratio*in_channels)

        self.conv1 = nn.Conv2d(cin, cin, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)

        self.conv2 = nn.Conv2d(cin, cin,
                               kernel_size = 3,
                               stride = 1,
                               padding= 1,
                               groups = cin,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(cin)

        self.conv3 = nn.Conv2d(cout,cout,kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)


    def forward(self, x):
        x1, x2, x3 = self.split(x)

        out = self.bn1(self.conv1(x3))
        out = self.bn2(self.conv2(out)) 
        
        c1, c2 = torch.chunk(out, 2,1)
        out = torch.cat([c2, x2], 1)
        
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = torch.cat([c1, x1,  out], dim=1)
        
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # right
        self.conv3 = nn.Conv2d(in_channels, mid_channels,kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels,kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels,kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)
        

    def forward(self, x):
        # left
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)), inplace=True)
        # right
        out2 = F.relu(self.bn3(self.conv3(x)), inplace=True)
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)), inplace=True)
        # concat
        out = torch.cat([out1, out2], 1)
        return out


class AugShuffleNetV2(nn.Module):
    def __init__(self, net_size,num_classes,split_ratio=(3/8)):
        super(AugShuffleNetV2, self).__init__()

        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24

        self.layer1 = self._make_layer(out_channels[0], num_blocks[0], 0.5)
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1], split_ratio)
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2], split_ratio)

        self.conv5 = nn.Conv2d(out_channels[2], out_channels[3],kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, out_channels, num_blocks,split_ratio):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels, split_ratio))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.max_pool2d(out, 3, stride=1, padding=1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        #out = nn.Dropout(0.1)(out)
        out = self.linear(out)
        return out






configs = {
    0.25: {
        'out_channels': (24, 48, 96, 1024),
        'num_blocks': (3, 7, 3)
    },
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1.0: {
        'out_channels': (120, 240, 480, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2.0: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}




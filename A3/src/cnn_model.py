import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, normalization='batch', num_filters_1=32, num_filters_2=64):
        super(BasicCNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.normalization = normalization

        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, num_filters_1, kernel_size=3, padding=1)

        # 根据归一化类型选择相应的层
        if normalization == 'batch':
            self.norm1 = nn.BatchNorm2d(num_filters_1)
        elif normalization == 'instance':
            self.norm1 = nn.InstanceNorm2d(num_filters_1)
        elif normalization == 'layer':
            self.norm1 = nn.GroupNorm(1, num_filters_1)
        elif normalization == 'group':
            self.norm1 = nn.GroupNorm(4, num_filters_1)
        else:  # 'none'
            self.norm1 = nn.Identity()

        # 第二个卷积块
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=3, padding=1)

        if normalization == 'batch':
            self.norm2 = nn.BatchNorm2d(num_filters_2)
        elif normalization == 'instance':
            self.norm2 = nn.InstanceNorm2d(num_filters_2)
        elif normalization == 'layer':
            self.norm2 = nn.GroupNorm(1, num_filters_2)
        elif normalization == 'group':
            self.norm2 = nn.GroupNorm(4, num_filters_2)
        else:  # 'none'
            self.norm2 = nn.Identity()

        # 全连接层
        self.fc1 = nn.Linear(num_filters_2 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class DeepCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, normalization='batch'):
        super(DeepCNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.normalization = normalization

        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.apply_norm1 = self._get_norm_layer(32)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.apply_norm2 = self._get_norm_layer(64)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.apply_norm3 = self._get_norm_layer(128)

        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.apply_norm4 = self._get_norm_layer(256)

        # 全连接层
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def _get_norm_layer(self, num_features):
        if self.normalization == 'batch':
            return nn.BatchNorm2d(num_features)
        elif self.normalization == 'instance':
            return nn.InstanceNorm2d(num_features)
        elif self.normalization == 'layer':
            return nn.GroupNorm(1, num_features)
        elif self.normalization == 'group':
            return nn.GroupNorm(min(32, num_features // 8), num_features)
        else:  # 'none'
            return nn.Identity()

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.apply_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.apply_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 第三个卷积块
        x = self.conv3(x)
        x = self.apply_norm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 第四个卷积块
        x = self.conv4(x)
        x = self.apply_norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, normalization='batch'):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = self._get_norm_layer(out_channels, normalization)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self._get_norm_layer(out_channels, normalization)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                self._get_norm_layer(out_channels, normalization)
            )

    def _get_norm_layer(self, num_features, normalization):
        if normalization == 'batch':
            return nn.BatchNorm2d(num_features)
        elif normalization == 'instance':
            return nn.InstanceNorm2d(num_features)
        elif normalization == 'layer':
            return nn.GroupNorm(1, num_features)
        elif normalization == 'group':
            return nn.GroupNorm(min(32, num_features // 8), num_features)
        else:  # 'none'
            return nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, dropout_rate=0.5, normalization='batch'):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._get_norm_layer(16, normalization)

        # 残差块
        self.layer1 = self._make_layer(16, 2, stride=1, normalization=normalization)
        self.layer2 = self._make_layer(32, 2, stride=2, normalization=normalization)
        self.layer3 = self._make_layer(64, 2, stride=2, normalization=normalization)

        # 全连接层
        self.linear = nn.Linear(64, 10)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def _get_norm_layer(self, num_features, normalization):
        if normalization == 'batch':
            return nn.BatchNorm2d(num_features)
        elif normalization == 'instance':
            return nn.InstanceNorm2d(num_features)
        elif normalization == 'layer':
            return nn.GroupNorm(1, num_features)
        elif normalization == 'group':
            return nn.GroupNorm(min(32, num_features // 8), num_features)
        else:  # 'none'
            return nn.Identity()

    def _make_layer(self, out_channels, num_blocks, stride, normalization):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride, normalization))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


# 模型集成类
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # 获取所有模型的预测结果
        outputs = [model(x) for model in self.models]

        # 对结果进行平均
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)

        return ensemble_output

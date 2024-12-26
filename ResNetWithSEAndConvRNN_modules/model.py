import torch
from torch import nn
from torchvision import models


class ResNetWithSEAndConvRNN(nn.Module):
    def __init__(self):
        super(ResNetWithSEAndConvRNN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # 修改第一层卷积
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 去掉初始的 MaxPool 层
        
        num_features = self.resnet.fc.in_features

        # 替换全连接层
        self.resnet.fc = nn.Sequential(
            nn.Linear(256, 128),  # ConvLSTM 输出的特征维度是 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # SVHN 有 10 个类别
        )

        # 添加 SE 模块
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # 添加卷积式 RNN (ConvLSTM)
        self.conv_rnn = ConvLSTM(
            input_dim=512,  # 对应 SE4 输出的通道数
            hidden_dim=256,  # RNN 隐藏状态维度
            kernel_size=(3, 3),
            num_layers=1,  # 单层 RNN
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

    def forward(self, x):
        # 通过 ResNet 的前几层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x = self.resnet.layer1(x)
        x = self.se1(x)

        x = self.resnet.layer2(x)
        x = self.se2(x)

        x = self.resnet.layer3(x)
        x = self.se3(x)

        x = self.resnet.layer4(x)
        x = self.se4(x)

        # ConvLSTM 处理，输入需要有时间序列维度，扩展维度模拟时间序列
        x = x.unsqueeze(1)  # (batch, seq_len=1, channels, height, width)
        _, last_states = self.conv_rnn(x)
        x = last_states[0][0]  # (batch, channels, height, width)

        # 平均池化，将特征图变为 2D
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling (batch, channels)

        # 全连接层
        x = self.resnet.fc(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        y = self.fc1(y)
        y = torch.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)  # Reshape to (b, c, 1, 1)
        return x * y


class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state

        combined = torch.cat([x, h_cur], dim=1)  # Concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """ConvLSTM Module"""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # (time, batch, channels, height, width)

        b, _, c, h, w = x.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], [h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_height, image_width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append((
                torch.zeros(batch_size, self.hidden_dim, image_height, image_width, device=self.cell_list[0].conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, image_height, image_width, device=self.cell_list[0].conv.weight.device)
            ))
        return init_states

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from Utils.getData import Data
from torch.optim import Adam

# Define a ResNet-like architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x

class ResNetLike(nn.Module):
    def __init__(self, input_c, output):
        super(ResNetLike, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_c, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = ResidualBlock(64, 64)
        self.resblock2 = ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def main():
    BATCH_SIZE = 32
    EPOCH = 15
    LEARNING_RATE = 0.001
    folds = [1, 2, 3, 4, 5]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_aug_loader = DataLoader(Data(augmented=f'C:/Users/ASUS/Pictures/Augmented Images/Augmented Images/FOLDS_AUG/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    train_ori_loader = DataLoader(Data(original=f'C:/Users/ASUS/Pictures/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Train']), batch_size=BATCH_SIZE, shuffle=True)
    vali_loader = DataLoader(Data(original=f'C:/Users/ASUS/Pictures/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Valid']), batch_size=BATCH_SIZE, shuffle=False)

    model = ResNetLike(input_c=3, output=6)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_vali_all = [], []
    for epoch in range(EPOCH):
        train_loss = 0
        vali_loss = 0
        model.train()
        for batch, (src, trg) in enumerate(train_aug_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        for batch, (src, trg) in enumerate(train_ori_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch, (src, trg) in enumerate(vali_loader):
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                src = torch.permute(src, (0, 3, 1, 2))
                pred = model(src)
                loss = loss_function(pred, trg)
                vali_loss += loss.item()
        
        loss_train_all.append(train_loss / (len(train_aug_loader) + len(train_ori_loader)))
        loss_vali_all.append(vali_loss / len(vali_loader))
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / (len(train_aug_loader) + len(train_ori_loader))}, Validation Loss: {vali_loss / len(vali_loader)}')

        if (epoch + 1) % 15 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / (len(train_aug_loader) + len(train_ori_loader)),
            }, "./ResNetLike_" + str(epoch + 1) + ".pt")
    
    plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
    plt.plot(range(EPOCH), loss_vali_all, color="#3399e6", label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./training_resnetlike.png")

if __name__ == "__main__":
    main()

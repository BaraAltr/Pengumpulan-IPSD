import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from Utils.getData import Data
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

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
    LEARNING_RATE = 0.001
    folds = [1,2,3,4,5]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_loader = DataLoader(Data(original=f'C:/Users/ASUS/Pictures/Original Images/Original Images/FOLDS/', folds=folds, subdir=['Test']), batch_size=BATCH_SIZE, shuffle=True)

    model = ResNetLike(input_c=3, output=6)
    model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint = torch.load('ResNetLike_15.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_batch = checkpoint['loss']

    prediction, ground_truth, probabilities = [], [], []
    with torch.no_grad():
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE) 
            src = torch.permute(src, (0, 3, 1, 2))  

            pred = model(src) 
            probabilities.extend(pred.detach().cpu().numpy()) 
            prediction.extend(torch.argmax(pred, dim=1).detach().cpu().numpy()) 
            ground_truth.extend(torch.argmax(trg, dim=1).detach().cpu().numpy())  

    classes = ('Chikenpox', 'Cowpox', 'Healty', 'HFMD', 'Measles', 'Monkeypox')

    cf_matrix = confusion_matrix(ground_truth, prediction)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')

    print("accuracy score = ", accuracy_score(ground_truth, prediction))
    print("precision score = ", precision_score(ground_truth, prediction, average='weighted'))
    print("recall score = ", recall_score(ground_truth, prediction, average='weighted'))
    print("f1 score score = ", f1_score(ground_truth, prediction, average='weighted'))

    ground_truth_bin = label_binarize(ground_truth, classes=list(range(len(classes))))
    probabilities = np.array(probabilities)

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(ground_truth_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc_auc.png')

if __name__ == "__main__":
    main()
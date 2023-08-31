import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim

import os
from PIL import Image
import argparse


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {c: int(c) for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.imgs.append((path, self.class_to_idx[c]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class Custom_model(nn.Module):
    def __init__(self):
        super(Custom_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)  # Adjusted input size
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 50)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        predicted_label = self.fc2(x)
        return predicted_label


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 50)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=50, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=50, bias=True))
    elif selection =='custom':
        model = Custom_model()
    return model


def train(net1, labeled_loader, optimizer, criterion):
    net1.train()
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        outputs = net1(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',  type=str,  default='False')
    parser.add_argument('--student_abs_path',  type=str,  default='./')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning'))

    batch_size = 32
    if args.test == 'False':
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = CustomDataset(root = './data/Supervised_Learning/labeled', transform = train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dataset = CustomDataset(root = './data/Supervised_Learning/val', transform = test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else :
        test_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    model_name = 'resnet' # e.g., 'resnet', 'vgg', 'mobilenet', 'custom'

    if torch.cuda.is_available():
        model = model_selection(model_name).cuda()
    else:
        model = model_selection(model_name)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    epoch = 50
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    best_result = 0
    if args.test == 'False':
        assert params < 7.0, "Exceed the limit on the number of model parameters"
        for e in range(0, epoch):
            train(model, labeled_loader, optimizer, criterion)
            tmp_res = test(model, val_loader)
            print('{}th performance, res : {}'.format(e, tmp_res))
            if best_result < tmp_res:
                best_result = tmp_res
                torch.save(model.state_dict(),  os.path.join('./logs', 'Supervised_Learning', 'best_model.pt'))
        print('Final performance {} - {}'.format(best_result, params))

    else:
        #This part is used to evaluate.
        #Do not edit this part!
        dataset = CustomDataset(root = '/data/23_1_ML_challenge/Supervised_Learning/test', transform = test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt'), map_location=torch.device('cuda')))
        res = test(model, test_loader)
        print(res, ' - ' , params)

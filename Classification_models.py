import torch
from matplotlib import pyplot as plt
from torch import nn, optim
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from matplotlib.ticker import MaxNLocator
from torchsummary import summary


import numpy as np
from PIL import Image
import os
import csv
import argparse
import time
 
# 超参数
batch_size = 32  # 批大小
learning_rate = 0.0001  # 学习率
epochs = 100  # 迭代次数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 清空cuda缓存
# fuser -k /dev/nvidia*
# 查看内存使用情况
# watch -n 10 nvidia-smi
torch.cuda.empty_cache()
print(device)


def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    parser.add_argument('-o', '--output', help='The dir path of output matrix', type=str, required=True)
    parser.add_argument('-t', '--type', help='The type of dataset(sard,realdata)', type=str, required=True)
    parser.add_argument('-m', '--model', help='The type of model(sard,realdata)', type=str, required=True)

    args = parser.parse_args()
    return args

# 均将图片转为三通道
sard_labels = {'NoVul':0,'Vul':1}
class MyData_RGB(Dataset):
    def __init__(self, root, transform=None):
        self.label_name = {'NoVul':0,'Vul':1}
        self.data_info = self.get_img_info(root)
        self.transform = transform

    def __getitem__(self,index):
        path_img,label = self.data_info[index]
        img = Image.open(path_img)
        # 灰度图转化为RGB
        # img = Image.open(path_img).convert('RGB')
        # 【单通道图】 转化为 【普通三通道图】
        img_numpy = np.array(img)
        img = np.array([img_numpy,img_numpy,img_numpy])
        img = np.transpose(img,(1,2,0))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label
 
    def __len__(self):
        return len(self.data_info)
    
    @staticmethod
    def get_img_info(root):
        data_info = list()
        for root, dirs, _ in os.walk(root):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'),img_names))
                
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # print(sub_dir)
                    label = sard_labels[sub_dir]
                    data_info.append((path_img, int(label)))         
        return data_info
    
    
def get_metric(y_pred, test_Y):
    f1 = f1_score(y_true=test_Y, y_pred=y_pred)
    precision = precision_score(y_true=test_Y, y_pred=y_pred)
    recall = recall_score(y_true=test_Y, y_pred=y_pred)
    accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

    TP = np.sum(np.multiply(test_Y, y_pred))
    FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
    FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
    TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (TP + FN)
    # print(FPR)
    result = [f1, precision, recall, accuracy, TPR, FPR, TNR, FNR]
    return result



def main():
    args = parse_options()
    if args.input[-1] == '/':
        in_path = args.input
    else:
        in_path = args.input + '/'
    out_path = args.output
    dataset_type = args.type
    model_type = args.model
    if out_path[-1] == '/':
        out_path += model_type + '/' + dataset_type + '/'
    else:
        out_path += '/' + model_type + '/' + dataset_type + '/'
    folder = os.path.exists(out_path)
    if not folder:
        os.makedirs(out_path)

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
    # transform = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize([0.5], [0.5])])


    if dataset_type == 'sard':
        root = in_path + 'sard/'
        dataset = MyData_RGB(root, transform)
    elif dataset_type == 'reveal':
        root = in_path + 'reveal/'
        dataset = MyData_RGB(root, transform)
    elif dataset_type == 'realdata':
        root_ffmpeg = in_path + 'ffmpeg/'
        dataset_ffmpeg = MyData_RGB(root_ffmpeg, transform)
        root_qemu = in_path + 'qemu/'
        dataset_qemu = MyData_RGB(root_qemu, transform)
        dataset = dataset_ffmpeg + dataset_qemu
    else:
        print('dataset type error!')
    num = len(dataset)
    train_size = int(num * 0.8)
    val_size = int(num * 0.1)
    test_size = num - train_size - val_size
    print(num, train_size, val_size, test_size)
    trainData, valData, testData = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True)
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=batch_size, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)

    # 选择模型
    # input_size = (3,98,98) dim=2
    if model_type == 'alexnet':
        # (3,224,224) 1000
        model = models.alexnet()
        model.features[0] = nn.Conv2d(3, 64, 11, stride=2, padding=8, bias=False)
        model.classifier[4] = nn.Linear(4096,512)
        model.classifier[6] = nn.Linear(512,2)
    elif model_type == 'squeeze':
        model = models.squeezenet1_1(num_classes=2)
        model.features[0] = nn.Conv2d(3, 64, 3, stride=1, padding=6)
    elif model_type == 'vgg':
        model = models.vgg16()
        model.features[0] = nn.Conv2d(3, 64, 3, stride=1, padding=8)
        model.features[4] = nn.Sequential()
        model.classifier[3] = nn.Linear(4096,512)
        model.classifier[6] = nn.Linear(512,2)
    elif model_type == 'mobilev2':
        model = models.mobilenet_v2()
        model.features[0][0] = nn.Conv2d(3, 32, 3, stride=1, padding=8)
        model.classifier = nn.Linear(1280,2)
    elif model_type == 'mobilev3l':
        model = models.mobilenet_v3_large()
        model.features[0][0] = nn.Conv2d(3, 16, 3, stride=1, padding=8)
        model.classifier[0] = nn.Linear(960,128)
        model.classifier[3] = nn.Linear(128,2)
    elif model_type == 'mobilev3s':
        model = models.mobilenet_v3_small()
        model.features[0][0] = nn.Conv2d(3, 16, 3, stride=1, padding=8)
        model.classifier[0] = nn.Linear(576,64)
        model.classifier[3] = nn.Linear(64,2)
    elif model_type == 'resnet':
        model = models.resnet18()
        model.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=10, bias=False)
        model.fc = nn.Linear(512,2)
    elif model_type == 'densenet':
        model = models.densenet121()
        model.features.conv0 = nn.Conv2d(3, 64, 7, stride=1, padding=10)
        model.classifier = nn.Linear(1024,2)
    # elif model_type == 'resnet50':
    #     model = models.resnet50()
    #     model.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=10, bias=False)
    #     model.fc = nn.Linear(2048,2)
    elif model_type == 'shuffle':
        model = models.shufflenet_v2_x1_0(num_classes=2)
        model.conv1 = nn.Conv2d(3, 24, 3, stride=1, padding=8)
    elif model_type == 'mnasnet':
        model = models.mnasnet1_0(num_classes=2)
        model.layers[0] = nn.Conv2d(3, 32, 3, stride=1, padding=8)
    elif model_type == 'resnext':   
        model = models.resnext50_32x4d()
        model.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=10, bias=False)
        model.fc = nn.Linear(2048,2)
    # elif model_type == 'vggbn':
    #     model = models.vgg16_bn()
    #     model.features[0] = nn.Conv2d(3, 64, 3, stride=1, padding=8)
    #     model.features[6] = nn.Sequential()
    #     model.classifier[3] = nn.Linear(4096,512)
    #     model.classifier[6] = nn.Linear(512,2)
    # elif model_type == 'densenet':
    #     model = models.densenet161()
    #     model.features.conv0 = nn.Conv2d(3, 96, 7, stride=1, padding=10)
    #     model.classifier = nn.Linear(2208,2)
    # elif model_type =='goolenet':
    #     model = models.googlenet()
    #     model.Conv1 = nn.Conv2d(3, 64, 7, stride=1,padding=10)
    #     model.fc = nn.Linear(1024,2)
    # elif model_type =='inception':
    #     model = models.inception_v3()
    #     model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, 3, stride=1,padding=8)
    #     model.fc = nn.Linear(2048,2)
    elif model_type == 'resnet_w':
        model = models.wide_resnet50_2()
        model.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=10, bias=False)
        model.fc = nn.Linear(2048,2)
    else:
        print(" Model Type Error!")

    # if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型
    
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 选用交叉熵函数作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型并存储训练时的指标
    epoch = 1
    index = 1
    history = {'Train Loss': [],
            'Val Loss': [],
            'Train Acc': [],
            'Val Acc': []}
    csv_data = [[] for i in range((epochs+2)*2)]
    csv_data[0] = ['Epoch', 'F1', 'Precision', 'Recall', 'Accuracy', 'TPR', 'FPR', 'TNR', 'FNR', 'Loss', 'Acc', 'Train_time','Val_time']
    for epoch in range(1, epochs+1):
        processBar = tqdm(trainDataLoader, unit='step')
        model.train(True)
        train_loss, train_correct = 0, 0
        for step, (train_imgs, labels) in enumerate(processBar):
            train_imgs = train_imgs.to(device)
            labels = labels.to(device)
            model.zero_grad()  # 梯度清零
            outputs = model(train_imgs)  # 输入训练集
            loss = criterion(outputs, labels)  # 计算损失函数
            predictions = torch.argmax(outputs, dim=1)  # 得到预测值
            correct = torch.sum(predictions == labels)
            accuracy = correct / labels.shape[0]  # 计算这一批次的正确率
            loss.backward()  # 反向传播
            optimizer.step()  # 更新优化器参数
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %  # 可视化训练进度条设置
                                    (epoch, epochs, loss.item(), accuracy.item()))
    
            # 记录下训练的指标
            train_loss = train_loss + loss
            train_correct = train_correct + correct
    
            # 当所有训练数据都进行了一次训练后，在验证集进行验证
            if step == len(processBar) - 1:
                # Val
                val_correct, totalLoss = 0, 0
                model.eval()  # 固定模型的参数并在测试阶段不计算梯度
                pred_np = np.array([])
                label_np = np.array([])
                with torch.no_grad():
                    for val_imgs, val_labels in valDataLoader:
                        val_imgs = val_imgs.to(device)
                        val_labels = val_labels.to(device)
                        val_outputs = model(val_imgs)
                        val_loss = criterion(val_outputs, val_labels)
                        predictions = torch.argmax(val_outputs, dim=1)
    
                        totalLoss += val_loss
                        val_correct += torch.sum(predictions == val_labels)
                        pred_np = np.append(pred_np, predictions.cpu().numpy())
                        label_np = np.append(label_np, val_labels.cpu().numpy())

                    train_accuracy = train_correct / len(trainDataLoader.dataset)
                    train_loss = train_loss / len(trainDataLoader)  # 累加loss后除以步数即为平均loss值
    
                    val_accuracy = val_correct / len(valDataLoader.dataset)  # 累加正确数除以样本数即为验证集正确率
                    val_loss = totalLoss / len(valDataLoader)  # 累加loss后除以步数即为平均loss值
    
                    history['Train Loss'].append(train_loss.item())  # 记录loss和acc
                    history['Train Acc'].append(train_accuracy.item())
                    history['Val Loss'].append(val_loss.item())
                    history['Val Acc'].append(val_accuracy.item())
                    metric_result = get_metric(pred_np, label_np)
                    metric_result.append(val_loss.item())
                    metric_result.append(val_accuracy.item())

                    csv_data[index].append(str(epoch)+'_val')
                    csv_data[index].extend(metric_result)
                    index += 1

                # Test
                # tst_correct, totalLoss = 0, 0
                model.eval()  # 固定模型的参数并在测试阶段不计算梯度
                pred_np = np.array([])
                label_np = np.array([])
                with torch.no_grad():
                    for test_imgs, test_labels in testDataLoader:
                        test_imgs = test_imgs.to(device)
                        test_labels = test_labels.to(device)
                        tst_outputs = model(test_imgs)
                        predictions = torch.argmax(tst_outputs, dim=1)
                        pred_np = np.append(pred_np, predictions.cpu().numpy())
                        label_np = np.append(label_np, test_labels.cpu().numpy())

                    metric_result = get_metric(pred_np, label_np)

                    csv_data[index].append(str(epoch)+'_test')
                    csv_data[index].extend(metric_result)
                    index += 1
    
                    processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f" %
                                            (epoch, epochs, train_loss.item(), train_accuracy.item(), val_loss.item(),
                                                val_accuracy.item()))
        processBar.close()
        file = out_path + 'result.csv'
        with open(file, 'w', newline='') as f:
            csvfile = csv.writer(f)
            csvfile.writerows(csv_data)
    # 对测试Loss进行可视化
    plt.plot(history['Val Loss'], color='red', label='Val Loss')
    plt.plot(history['Train Loss'], label='Train Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim([0, epoch])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Loss')
    plt.title('Train and Val LOSS')
    plt.legend(loc='upper right')
    plt.savefig(out_path + 'LOSS')
    plt.show()
    
    # 对测试准确率进行可视化
    plt.plot(history['Val Acc'], color='red', label='Val Acc')
    plt.plot(history['Train Acc'], label='Train Acc')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.xlim([0, epoch])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Accuracy')
    plt.title('Train and Val ACC')
    plt.legend(loc='lower right')
    plt.savefig(out_path + 'ACC')
    plt.show()
    
    # torch.save(model, '../model/realdata.pth')

if __name__ == '__main__':
    main() 

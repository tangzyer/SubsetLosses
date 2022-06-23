import matplotlib.pyplot as plt
import re
import pandas as pd

def plot_text():
    epoch_num = 0
    Loss = []
    True_Loss = []
    Accuracy = []

    with open('nohup.out', 'r') as f:
            for line in f:
                text = line.strip('\n').split(',')[0]
                num = float(re.findall(r'-?\d+\.?\d*e?-?\d*?', text)[-1])
                if 'Pseudo' not in text:
                    if 'True Loss' in text:
                        epoch_num += 1
                        True_Loss.append(num)
                    elif 'Loss' in text:
                        Loss.append(num)
                    else:
                        Accuracy.append(num)

    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    x3 = range(0, epoch_num)
    y1 = Accuracy
    y2 = Loss
    y3 = True_Loss
    plt.subplot(3, 1, 1)
    plt.plot(x1, y2, 'o-')
    plt.title('Validation indicators vs. epoches')
    plt.ylabel('Train loss')
    plt.subplot(3, 1, 2)
    plt.plot(x2, y3, '.-')
    plt.xlabel('epochs')
    plt.ylabel('True loss')
    plt.subplot(3, 1, 3)
    plt.plot(x3, y1, '.-')
    plt.xlabel('epochs')
    plt.ylabel('Valid Accuracy')
    plt.savefig('./logs/l1loss.png')


def plot_csv(csv_path, label):
    df = pd.read_csv(csv_path)
    legend = []
    x = df.index.values
    plt.subplot(2, 1, 1)
    plt.ylabel('loss')
    for col in df.columns:
        if 'Loss' in col:
            plt.plot(x, df[col], '.-', linewidth=1)
            legend.append(col)
    plt.legend(legend)
    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    plt.ylabel('Valid Accuracy')
    for col in df.columns:
        if 'Loss' not in col and 'Unnamed' not in col:
            plt.plot(x, df[col], '.-', linewidth=2)
    plt.savefig('./logs/'+label+'.png')


    
plot_csv('logs/l1loss22_55_06_21.csv', 'l1loss_2')
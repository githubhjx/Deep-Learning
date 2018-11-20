#  This is a Pain Dataset cut program
#  Created by JiaxuHan at 2018/11/19

import os
import glob
import shutil


record_path = '/home/s2/data/Pain/records.txt'
label_path = '/home/s2/data/Pain/label.txt'
img_path = '/home/s2/data/Pain/79/'
out_path = '/home/s2/data/Pain/Cuted/'


def getrecord(path):
    records = []
    file = open(path)
    file = file.readlines()
    for i in range(len(file)):
        f = file[i].split()
        ft = list(map(int, f))
        records.append(ft)
    return records


def getlabel(path):
    labels = []
    file = open(path)
    file = file.readlines()
    for i in range(len(file)):
        f = file[i].split()
        ft = list(map(int, f))
        labels.append(ft)
    return labels


def cut_img(path):

    num_dir = 0
    labels = []

    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort()

    record = getrecord(record_path)
    label = getlabel(label_path)

    for idx, folder in enumerate(cate):

        L = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        temp = glob.glob(folder + '/*.png')
        temp.sort()
        for i, j in zip(range(0, len(record[idx]), 2), range(1, len(record[idx]), 2)):
            label_temp = []
            start = record[idx][i]-1
            end = record[idx][j]-1
            if start<end:
                cut_id = list(range(start, end+1))
            else:
                cut_id = list(range(start, end-1, -1))

            path_temp = out_path + str(num_dir) + '/'
            if os.path.exists(path_temp):
                shutil.rmtree(path_temp)
                os.mkdir(path_temp)
            else:
                os.mkdir(path_temp)

            for id, imgid in enumerate(cut_id):
                shutil.copyfile(temp[imgid], path_temp + str(id) + '.png')
                label_temp.append(label[idx][imgid])
            labels.append(label_temp)
            num_dir = num_dir + 1

    f = open(out_path + 'label.txt', 'w')
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            f.write(str(labels[i][j]))
            f.write(' ')
        f.write('\n')
    f.close()
    print("Image has been reprocessed!")


getlabel(label_path)
cut_img(img_path)

import os
import sys
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset
from mydatasets import ImageFolder

sys.setrecursionlimit(100000)

pre_model = 'LightCNN_9Layers_checkpoint.pth.tar'

tr_path = ''

gallery_path = ''
val_path = ''

save_model = ''
loss_log_path = ''
acc_log_path = ''

epochs = 100
BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(description='NIR-VIS')
    parser.add_argument('--pre_model', default=pre_model)
    parser.add_argument('--train_path', default=tr_path)
    parser.add_argument('--val_path', default=val_path)
    parser.add_argument('--gallery_path', default=gallery_path)
    parser.add_argument('--save_path', default=save_model)
    parser.add_argument('--loss_log_path', default=loss_log_path)
    parser.add_argument('--acc_log_path', default=acc_log_path)
    parser.add_argument('--batch_size', default=BATCH_SIZE)

    args = parser.parse_args()
    return args


class Mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, f_type=1):
        super(Mfm, self).__init__()
        self.out_channels = out_channels
        if f_type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class Group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Group, self).__init__()
        self.conv_a = Mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = Mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = Mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class Network(nn.Module):
    def __init__(self, num_classes=363):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            Mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Group(192, 128, 3, 1, 1),
            Group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = Mfm(8 * 8 * 128, 128, f_type=0)
        self.fc_n_1 = Mfm(8 * 8 * 128, 128, f_type=0)
        self.fc_v_1 = Mfm(8 * 8 * 128, 128, f_type=0)

        # self.fc2 = nn.Linear(128, num_classes)
        self.fc_n_2 = nn.Linear(2*128, num_classes)
        self.fc_v_2 = nn.Linear(2*128, num_classes)

    def forward(self, x_n, x_v):
        x_n = self.features(x_n)
        x_v = self.features(x_v)

        # input unique branch
        x_n = x_n.view(x_n.size(0), -1)
        x_v = x_v.view(x_v.size(0), -1)

        # feature share
        x_w_n = F.dropout(self.fc1(x_n), training=self.training)
        x_w_v = F.dropout(self.fc1(x_v), training=self.training)

        # feature unique
        x_p_n = F.dropout(self.fc_n_1(x_n), training=self.training)
        x_p_v = F.dropout(self.fc_v_1(x_v), training=self.training)

        # output
        output_1 = self.fc_n_2(torch.cat((x_w, x_p_n), 1))
        output_2 = self.fc_v_2(torch.cat((x_w, x_p_v), 1))

        return output_1, output_2, x_w_n, x_w_v


def network(**kwargs):
    model = Network(**kwargs)
    return model


# # tensorboard show
# dumpy_input = torch.randn(1, 1, 128, 128)
#
# with SummaryWriter(comment='my_net') as w:
#     w.add_graph(net, (dumpy_input, dumpy_input))

# get shared W and unique P_N, P_V


# get loss
def get_or4loss(net):
    for name, param in net.named_parameters():
        if name == 'fc1.filter.weight':
            w = param
        if name == 'fc_n_1.filter.weight':
            p_n = param.t()
        if name == 'fc_v_1.filter.weight':
            p_v = param.t()

    #  Orthogonal constraint
    loss_or = torch.mean(torch.norm(torch.matmul(p_n, w), 2)) + torch.mean(torch.norm(torch.matmul(p_v, w), 2))
    return loss_or


# params = net.named_parameters()
# (name, param) = params[0]
# print(name)
# print(param.grad)

# load pre_train parameters
def load_param(net, tar_model):
    # get pre_train parameters
    tar_model = torch.load(tar_model, map_location='cpu')
    pre_dict = tar_model['state_dict']

    # get net param filter diff layer
    model = net.state_dict()
    pre_dict = {k[7:]: v for k, v in pre_dict.items() if k[7:] in model}
    pre_dict = {k: v for k, v in pre_dict.items() if k[0:2] != 'fc'}

    # update net param use pre_train param
    model.update(pre_dict)

    # load model param to net
    net.load_state_dict(model)


def cal_distance(embeddings1, embeddings2):
    # l2-norm and distance
    feature1 = F.normalize(embeddings1)
    feature2 = F.normalize(embeddings2)
    dist = feature1.mm(feature2.t())
    return dist


def train(model, device, train_loader, optimizer, criterion, epoch, args, writer):
    start_epoch_time = time.time()

    # set graph mode
    model.train()

    # VIS iter
    iter_vis = iter(train_loader['VIS'])
    losses = []
    for batch_idx, (images_nir, target) in enumerate(train_loader['NIR']):
        start_batch_time = time.time()
        images_vis, v_target = next(iter_vis)

        # data to gpu
        images_nir, images_vis, target, v_target = \
            images_nir.to(device), images_vis.to(device), target.to(device), v_target.to(device)

        # run graph
        output1, output2, _, _ = model(images_nir, images_vis)

        # calculate loss
        or_loss = get_or4loss(model)
        cross_loss = criterion(output1, target) + criterion(output2, v_target)
        loss = or_loss + cross_loss

        # mean loss per epoch
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_batch_time = time.time()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tBatch: {} \tTime: {:.6f}'.format(
                epoch, (batch_idx + 1) * args.batch_size, args.batch_size * len(train_loader['NIR']),
                100. * (batch_idx / len(train_loader['NIR'])), loss.item(), batch_idx, end_batch_time-start_batch_time))

    end_epoch_time = time.time()
    writer.add_scalar(args.loss_log_path, np.mean(losses), epoch)
    print('Train Epoch: {} \tAverage Loss: {:.6f} \tTotal Time: {:.6f}'.format(epoch, np.mean(losses),
                                                                               end_epoch_time - start_epoch_time))


def test(model, device, test_loader, gallery_loader, epoch, args):
    start_test_time = time.time()
    model.eval()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    results = []
    dist_matrix = []
    for i, (gallery, glabel) in enumerate(gallery_loader):

        start_gallery_time = time.time()
        temp = []
        for batch_id, (image_nir, target) in enumerate(test_loader):
            gallery_temp = gallery
            gallery_temp = gallery_temp.expand(len(image_nir), gallery_temp.size(1), gallery_temp.size(2),
                                               gallery_temp.size(3))

            image_nir, gallery_temp = image_nir.to(device), gallery_temp.to(device)
            with torch.no_grad():
                _, _, nir_feature, gallery_feature = model(image_nir, gallery_temp)

            # nir_feature, gallery_feature = nir_feature.to('cpu'), gallery_feature.to('cpu')
            dist = cal_distance(nir_feature, gallery_feature)
            temp.extend(list(dist))
        dist_matrix.append(temp)

        end_time_per_gallery = time.time()
        print('Test gallery: {} \tTime: {:.6f}'.format(i+1, end_time_per_gallery-start_gallery_time))

    end_test_time = time.time()
    print('Test Total Time: {:.6f}'.format(end_test_time-start_test_time))

    # hash_matrix = np.transpose(dist_matrix)
    # predict_label = np.argmax(hash_matrix, axis=1)

    predict_labels = np.argmax(dist_matrix, axis=0)
    print(predict_labels)
    tar_label = []
    for idx, (image, label) in enumerate(test_loader):

        tar_label.extend(list(label.numpy()))

    results.extend(np.array(predict_labels) == np.array(tar_label))

    last_time = time.time()

    writer = SummaryWriter('accuracy')
    writer.add_scalar(args.acc_log_path, np.mean(results), epoch)
    writer.close()

    print('Epoch: %d, Rank-1 Test Accuracy: %f, Test Time:{:.6f}'.format(epoch + 1, np.mean(results),
                                                                         last_time - start_test_time))


def test_v2(model, device, test_loader, gallery_loader, epoch, args, writer):
    start_extract_time = time.time()

    # set graph mode
    model.eval()
    gallery_feature = []
    gallery_label = []
    probe_feature = []
    probe_label = []

    # get gallery feature
    for i, (gallery, g_label) in enumerate(gallery_loader):
        gallery = gallery.to(device)
        with torch.no_grad():
            _, _, _, g_feature = model(gallery, gallery)
            g_feature = F.normalize(g_feature)
            gallery_feature.extend(g_feature.tolist())
            gallery_label.extend(g_label)

    # get probe feature
    for j, (probe, p_label) in enumerate(test_loader):
        probe = probe.to(device)
        with torch.no_grad():
            _, _, p_feature, _ = model(probe, probe)
            p_feature = F.normalize(p_feature)
            probe_feature.extend(p_feature)
            probe_label.extend(p_label)

    end_extract_time = time.time()
    gf = torch.Tensor(gallery_feature).cuda()

    acc = 0
    # calculate rank-1 accuracy
    for k in range(len(probe_feature)):
        pf = probe_feature[k]
        diff = np.subtract(pf, gf)
        dist = np.sum(np.square(diff), 1)
        pred = torch.argmax(dist).item()
        if pred == gallery_label[k]:
            acc += 1

    end_cal_acc_time = time.time()
    time_e = end_extract_time - start_extract_time
    time_c = end_cal_acc_time - end_extract_time

    writer.add_scalar(args.acc_log_path, acc/len(probe_label), epoch)
    print('Epoch: %d, Rank-1 Test Accuracy: %f, Extract Feature Time:{:.6f}, Cal Acc Time:{:.6f}'.format
          (epoch + 1, acc/len(probe_label), time_e, time_c))


def main():

    args = parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make dataset
    tr_transform = {
        'NIR': transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'VIS': transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    val_transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize,
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    gallery_transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize,
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # make train loader
    datasets_train = {x: ImageFolder(os.path.join(args.train_path, x), tr_transform[x]) for x in ['NIR', 'VIS']}

    train_loaders = {x: torch.utils.data.DataLoader(datasets_train[x],
                                                    batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                    pin_memory=False) for x in ['NIR', 'VIS']}

    # val loader
    datasets_val = ImageFolder(os.path.join(args.val_path, 'NIR'), val_transform)

    val_loaders = torch.utils.data.DataLoader(datasets_val,
                                              batch_size=128, shuffle=False, num_workers=2, pin_memory=False)

    # gallery loader
    datasets_gallery = ImageFolder(os.path.join(args.gallery_path, 'VIS'), gallery_transform)

    gallery_loaders = torch.utils.data.DataLoader(datasets_gallery,
                                                  batch_size=128, shuffle=False, num_workers=2, pin_memory=False)

    model = network().to(device)
    load_param(model, args.pre_model)

    # model = torch.nn.DataParallel(model)
    # model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    writer = SummaryWriter()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loaders, optimizer, criterion, epoch, args, writer)
        test_v2(model, device, val_loaders, gallery_loaders, epoch, args, writer)
        if epoch % 100 == 0:
            torch.save(model, args.save_path + 'NIR_VIS_MODEL.pth.tar')
            torch.save(model.state_dict(), args.save_path + 'NIR_VIS_Param.pth.tar')

            print('Have been keep this model!')

            test_v2(model, device, val_loaders, gallery_loaders, epoch, args, writer)

    writer.close()

    # if args.save_path:
    #     torch.save(model, args.save_path + 'NIR_VIS_MODEL.pth.tar')
    #     torch.save(model.state_dict(), args.save_path + 'NIR_VIS_Param.pth.tar')
    #
    #     print('Have been keep this model!')


if __name__ == '__main__':
    main()

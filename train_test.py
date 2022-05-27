from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import random
import os
import visdom
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import shutil


from dataset import GA_Datas
from model import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default= 0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=0.0005, metavar='R',
                    help='weight decay') # 10e-5
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='gated_attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_num(Labels):
    p_num = 0
    n_num = 0
    for i in range(len(Labels)):
        if Labels[i]:
            p_num += 1
        else:
            n_num += 1

    return p_num, n_num


def train(epoch, model, optimizer, train_loader, train_num):
    model.train()
    train_loss = 0.
    train_right = 0.
    # lamda = 0.01
    b = 0.3
    for batch_idx, dict in enumerate(train_loader):
        data = dict["flow_x"]
        label = dict["flow_y"]
        # bag_label = label[0]
        adj = dict["graph"]
        bag_label = label
        if args.cuda:
            data, bag_label, adj = data.cuda(), bag_label.cuda(), adj.cuda()
        data, bag_label, adj = Variable(data), Variable(bag_label), Variable(adj)

        # regularization_loss = 0
        # for param in model.parameters():
        #     regularization_loss += torch.sum(abs(param))

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        # loss, _ = model.calculate_objective(data, bag_label, adj)
        # loss = loss + lamda * regularization_loss
        orgin_loss, _ = model.calculate_objective(data, bag_label, adj)
        loss = (orgin_loss - b).abs() + b  # Flood Regularization
        # train_loss += loss.item() * len(label)
        train_loss += loss.item()
        right, _ = model.calculate_classification_error(data, bag_label, adj)
        train_right += right
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        # ExpLR.step()

    # calculate loss and error for epoch
    # train_loss /= train_num
    # train_right /= train_num
    train_loss /= len(train_loader)
    train_right /= len(train_loader)
    train_acc = 100. * train_right

    print('Epoch: {}, Loss: {:.4f}, Train right: {:.4f}'.format(epoch, train_loss, train_right), end=' ')

    return train_loss, train_acc


def test(model, test_loader, test_num):
    model.eval()
    test_loss = 0.
    test_right = 0.
    for batch_idx, batch_data in enumerate(test_loader):
        data = batch_data["flow_x"]
        label = batch_data["flow_y"]
        # bag_label = label[0]
        adj = batch_data["graph"]
        bag_label = label
        # instance_labels = label[1]
        if args.cuda:
            data, bag_label, adj = data.cuda(), bag_label.cuda(), adj.cuda()
        data, bag_label, adj = Variable(data), Variable(bag_label), Variable(adj)
        loss, attention_weights = model.calculate_objective(data, bag_label, adj)
        # test_loss += loss.item() * len(label)
        test_loss += loss.item()
        right, predicted_label = model.calculate_classification_error(data, bag_label, adj)
        test_right += right
        # print(predicted_label)

        # if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
        #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        #     instance_level = list(zip(instance_labels.numpy()[0].tolist(),
        #                          np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
        #
        #     print('\nTrue Bag Label, Predicted Bag Label: {}\n'
        #           'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    # test_right /= test_num
    # test_loss /= test_num
    test_right /= len(test_loader)
    test_loss /= len(test_loader)
    test_acc = 100. * test_right

    # print('Epoch: {}, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, test_loss.cpu().numpy()[0], test_error))
    # print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))
    print('Test Set, Loss: {:.4f}, Test right: {:.4f}'.format(test_loss, test_right))

    return test_loss, test_acc


def metricsall(model, test_loader):
    model.eval()
    y_actual = []
    y_score = []
    y_pred = []
    test_loss = 0.
    test_right = 0.
    for batch_idx, batch_data in enumerate(test_loader):
        data = batch_data["flow_x"]
        label = batch_data["flow_y"]
        # bag_label = label[0]
        adj = batch_data["graph"]
        bag_label = label
        # instance_labels = label[1]
        if args.cuda:
            data, bag_label, adj = data.cuda(), bag_label.cuda(), adj.cuda()
        data, bag_label, adj = Variable(data), Variable(bag_label), Variable(adj)

        loss, attention_weights = model.calculate_objective(data, bag_label, adj)
        # test_loss += loss.item() * len(label)
        test_loss += loss.item()

        right, predicted_score = model.calculate_classification_error(data, bag_label, adj)
        predicted_label = predicted_score.max(1)[1]
        test_right += right

        # print(label.shape[0])
        for i in range(label.shape[0]):
            y_actual.append(int(label[i]))
            y_score.append(predicted_score[i][1].item())
            y_pred.append(predicted_label[i].item())

    # test_right /= test_num
    # test_loss /= test_num
    test_right /= len(test_loader)
    test_loss /= len(test_loader)
    test_acc = 100. * test_right

    # print('Epoch: {}, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, test_loss.cpu().numpy()[0], test_error))
    # print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))
    print('Test Set, Loss: {:.4f}, Test right: {:.4f}'.format(test_loss, test_right))
    myacc = test_acc
    # print(y_actual, y_score, y_perd)
    fpr, tpr, thresholds = metrics.roc_curve(y_actual, y_score)
    # print(fpr, tpr, thresholds)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(y_actual, y_pred)
    # maxtrix = metrics.confusion_matrix(y_actual, y_pred)
    sen = metrics.recall_score(y_actual, y_pred)
    spe = metrics.recall_score(y_actual, y_pred, pos_label=0)
    print('auc:', auc)
    print('myacc:', myacc)

    return test_loss, test_acc, fpr, tpr, f1, sen, spe


if __name__ == "__main__":
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        # torch.manual_seed(args.seed)
        # random.seed(args.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    print('num-worker:', 2)
    data_path = "./Feature/"
    batchSize = 32
    print('learning rate:', args.lr, 'weight decay:', args.reg, 'batch size:', batchSize)

    k = 5
    accList = []
    senList = []
    speList = []
    f1List = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    PATH = './Model/aggcn/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    for i in range(k):
        start_time = datetime.datetime.now()
        num = str(i+1)
        trainData = torch.load(data_path + num + "-fold/da-traindata-" + num + ".pth")
        testData = torch.load(data_path + num + "-fold/sig-testdata-" + num + ".pth")
        train_num, test_num = len(trainData), len(testData)
        print(train_num, test_num)
        train_loader = data_utils.DataLoader(GA_Datas(trainData),
                                             batch_size=batchSize,
                                             shuffle=True,
                                             **loader_kwargs)

        test_loader = data_utils.DataLoader(GA_Datas(testData),
                                            # batch_size=batchSize,
                                            batch_size=test_num,
                                            shuffle=True,
                                            **loader_kwargs)

        print('Init Model')
        if args.model == 'gated_attention':
            # model = GatedAttention()
            # model = GCNbatch(512, 256, 100, 0)
            # model = GCNbatch(512, 256, 100, 50, 0)
            # model = GCNbatch(512, 256, 0, 128, 0)
            # model = GCNbatchPeaNew(512, 256, 0, 128, 0)
            # model = GatedAttention()
            # model = GCNbatchAttNet(512, 256, 0, 128, 0)
            model = GCNbatchAtten(512, 256, 0, 128, 0)
        if args.cuda:
            model.cuda()

        print('Model:', args.model)

        print('Fold Lr:', args.lr)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.reg) # , weight_decay=args.reg
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, verbose=True, patience=5)

        # viz = visdom.Visdom(env='server-true-gcn-attenadd-5k-SA')
        # viz = visdom.Visdom(env='server-true-gcn-5k-SA-08')
        # viz = visdom.Visdom(env='experiment-true-gcn')

        print('The ' + num + ' Fold, Start Training')
        bestAcc = 0.
        bestEpoch = 0
        lastAcc = 0.

        start_epoch = 1
        K_PATH = PATH + str(i + 1) + '-fold/'
        if not os.path.exists(K_PATH):
            os.makedirs(K_PATH)
        else:
            model_list = os.listdir(K_PATH)
            model_num = len(model_list)
            if model_num == args.epochs:
                start_epoch = model_num + 1
                M_PATH = K_PATH + str(i + 1) + 'f-' + str(model_num) + '-gcn-attenadd'
                print('Load model:', M_PATH)
                model.load_state_dict(torch.load(M_PATH))
                test_loss, test_acc, fpr, tpr, f1, sen, spe = metricsall(model, test_loader)
                lastAcc = test_acc
                f1List.append(f1)
                senList.append(sen)
                speList.append(spe)

                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
            else:
                print('now epoch:', model_num)
                shutil.rmtree(K_PATH)
                os.makedirs(K_PATH)

        print('start:', start_epoch)


        for epoch in range(start_epoch, args.epochs + 1):

            # if epoch % 10 == 0:
            #     for p in optimizer.param_groups:
            #         p['lr'] *= 0.9

            train_loss, train_acc = train(epoch, model, optimizer, train_loader, train_num)
            # valid_loss, valid_acc = valid(model, valid_loader, valid_num)
            # scheduler.step()
            torch.save(model.state_dict(), K_PATH + str(i + 1) + 'f-' + str(epoch) + '-gcn-attenadd',
                       _use_new_zipfile_serialization=False)
            if epoch != args.epochs:
                test_loss, test_acc = test(model, test_loader, test_num)
            else:
                test_loss, test_acc, fpr, tpr, f1, sen, spe = metricsall(model, test_loader)
                lastAcc = test_acc
                f1List.append(f1)
                senList.append(sen)
                speList.append(spe)

                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

            scheduler.step(test_loss)
            # scheduler.step(test_acc)
            # bri_loss, bri_acc = test(model, bridge_loader, bridge_num)
            if test_acc > bestAcc:
                bestAcc = test_acc
                bestEpoch = epoch
            acc_win = 'accuracy-fold' + num
            loss_win = 'loss-fold' + num

            '''
            if epoch == 1:
                viz.line([[train_loss, test_loss]], [epoch], win=loss_win, opts=dict(title=loss_win, legend=['train_loss', 'test_loss']))
                viz.line([[train_acc, test_acc]], [epoch], win=acc_win, opts=dict(title=acc_win, legend=['train_acc', 'test_acc']))
            else:
                viz.line([[train_loss, test_loss]], [epoch], win=loss_win, update='append')
                viz.line([[train_acc, test_acc]], [epoch], win=acc_win, update='append')
            
            if epoch == 1:
                viz.line([train_loss], [epoch], win='train_loss', opts=dict(title='train loss'))
                viz.line([train_acc], [epoch], win='train_acc', opts=dict(title='train acc'))

                viz.line([test_loss], [epoch], win='test_loss', opts=dict(title='test loss'))
                viz.line([test_acc], [epoch], win='test_acc', opts=dict(title='test acc'))

                # viz.line([bri_loss], [epoch], win='bri_loss', opts=dict(title='bridge loss'))
                # viz.line([bri_acc], [epoch], win='bri_acc', opts=dict(title='bridge acc'))
            else:
                viz.line([train_loss], [epoch], win='train_loss', update='append')
                viz.line([train_acc], [epoch], win='train_acc', update='append')

                viz.line([test_loss], [epoch], win='test_loss', update='append')
                viz.line([test_acc], [epoch], win='test_acc', update='append')

                # viz.line([bri_loss], [epoch], win='bri_loss', update='append')
                # viz.line([bri_acc], [epoch], win='bri_acc', update='append')
            '''


        accList.append(lastAcc)
        print('The Best Accuracy of', num, 'fold is', bestAcc, 'When epoch:', bestEpoch)
        print('The Last Accuracy of', num, 'fold is', lastAcc)
        print('Finish')
        end_time = datetime.datetime.now()
        time = str((end_time - start_time).seconds)
        print('with ' + time + ' seconds, ' + 'finished in ' + str(datetime.datetime.now()).split('.')[0])


    avgAcc = np.mean(accList)
    print('The average accuracy of 5 fold is', avgAcc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    roc_data = "./roc/data/"
    if not os.path.exists(roc_data):
        os.makedirs(roc_data)
    np.save("./roc/data/fpr-gcn-attenadd-5k-SA.npy", mean_fpr)
    np.save("./roc/data/tpr-gcn-attenadd-5k-SA.npy", mean_tpr)

    print('Sensitivity:', np.mean(senList), 'Specificity:', np.mean(speList), 'F1-score:', np.mean(f1List), 'AUC:', mean_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    # plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    # plt.show()
    roc_img = "./roc/img/"
    if not os.path.exists(roc_img):
        os.makedirs(roc_img)
    plt.savefig("./roc/img/roc-gcn-attenadd-5k-SA.png")

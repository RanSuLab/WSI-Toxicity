import cv2
import shutil
import random
import math
import numpy as np
import os
import datetime


# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:

        distance = []
        for i in range(k):
            distance.append(distEclud(data, centroids[i]))
        '''
        diff = np.tile(data, (k,
                              1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        '''
        # print(distance)
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# 判断是否收敛
def converge(changed):
    for i in range(len(changed)):
        if np.any(changed[i] != 0):
            return True
    return False


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    # print(minDistIndices)

    newCentroids = []
    for i in range(k):
        img_sum = np.zeros((3, 256))
        for j, index in enumerate(minDistIndices):
            if i == index:
                img_sum += dataSet[j]
        if np.sum(minDistIndices == i) == 0:
            return np.ones(1), centroids
        newCentroids.append(img_sum / np.sum(minDistIndices == i))
    # print(newCentroids)
    '''
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values
    '''
    # 计算变化量
    # print(len(newCentroids), len(centroids))
    # changed = newCentroids - centroids
    changed = []
    for i in range(k):
        changed.append(newCentroids[i] - centroids[i])

    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)
    iteration = 1

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    # while np.any(changed != 0):
    while converge(changed):
        changed, newCentroids = classify(dataSet, newCentroids, k)
        iteration += 1

    # centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    # clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    clalist = calcDis(dataSet, newCentroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    # print(iteration)
    # print(minDistIndices)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        # cluster[j].append(dataSet[i])
        cluster[j].append(i+1)

    cluster.sort(key=lambda cluster: cluster[0])

    return centroids, cluster


# 创建数据集
def createDataSet(img_path):
    imlist = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')]
    imlist.sort(key=lambda x:int(x.split('.')[0].split('_')[-1]))
    dataSet = []
    for f in imlist:
        img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), -1)
        # img = cv2.imread(f)
        chans = cv2.split(img)
        features = np.zeros([len(chans), 256])
        for i, chan in enumerate(chans):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features[i] = hist.T
            # print(hist.shape)
            # features.extend(hist.T)
            '''
            if i == 0:
                features = hist.T
            else:
                features = np.hstack((features, hist.T))
            '''
        # print(features.shape)
        dataSet.append(features)
    return dataSet, imlist


def get_bags(patch_path, target_path):
    num = 0
    for svs in os.listdir(patch_path):
        whole_path = patch_path + svs + '/'
        for f in os.listdir(whole_path):
            num += 1
            file_path = whole_path + f + '/'
            svs_name = f.split('+')[0]
            target = target_path + svs + '/' + f + '/'
            if not os.path.exists(target):
                os.makedirs(target)

            k = 9
            dataset, pathlist = createDataSet(file_path)
            # print(pathlist)
            rate = 300 / len(pathlist)
            print('Cluster ', str(num), 'st image named ', f, end=' ')

            start_time = datetime.datetime.now()
            centroids, cluster = kmeans(dataset, k)
            for i in range(k):
                sam_num = int(math.ceil(len(cluster[i]) * rate))
                sam_patch = sorted(random.sample(cluster[i], sam_num))
                for j in range(sam_num):
                    source = file_path + svs_name + '_' + str(sam_patch[j]) + '.jpg'
                    # target = target_path + f + '/'
                    shutil.copy(source, target)

            bag_list = [os.path.join(target, f) for f in os.listdir(target) if f.endswith('.jpg')]
            if len(bag_list) > 300:
                del_list = random.sample(bag_list, len(bag_list) - 300)
                for d in range(len(del_list)):
                    os.remove(del_list[d])

            end_time = datetime.datetime.now()
            time = str((end_time - start_time).seconds)
            print('with ' + time + ' seconds, ' + 'finished ' + str(num) + ' in ' +
                  str(datetime.datetime.now()).split('.')[
                      0])

    print('Cluster Finish!')


if __name__ == '__main__':
    patch_path = './Patch/'
    target_path = './Bag/'
    get_bags(patch_path, target_path)

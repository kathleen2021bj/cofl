import numpy as np
import gzip
import os
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


def readFile(path):
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    print(f)
    first_ele = True
    matrix=[]
    for data in f.readlines():
        ## 去掉每行的换行符，"\n"
        data = data.strip('\n')
        #print("AAAA+"+data)
        ## 按照 空格进行分割。
        nums = data.split(' ')
        ## 添加到 matrix 中。
        if first_ele:
            ### 加入到 matrix 中 。
            #nums=nums.astype(np.float32)  np.float32(20140131.0)
            nums = [float(x) for x in nums]
            matrix = np.array(nums)
            matrix = matrix.astype(np.float32)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            #nums = nums.astype(np.float32)
            matrix = np.c_[matrix, nums]
            matrix = matrix.astype(np.float32)

    matrix = matrix.transpose()

    """
    a = []
    for x in range(0, 5610):
        result = [float(item) for item in matrix[x]]
        a.append(result)
    arr = np.array(a)
    """
    f.close()
    #print(matrix)
    #print(type(matrix))
    #print(matrix.dtype)
    #print(matrix.shape)
    return matrix


def readFileInt(path):
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    print(f)
    first_ele = True
    matrix=[]
    for data in f.readlines():
        ## 去掉每行的换行符，"\n"
        data = data.strip('\n')
        #print("AAAA+"+data)
        ## 按照 空格进行分割。
        nums = data.split(' ')
        matrix.append(int(nums[0]))
    #     len=int(nums[0])
    #     ## 添加到 matrix 中。
    #     """
    #     array_h = np.zeros((6), dtype='int')
    #     array_h[4]=1
    #     """
    #     if first_ele:
    #         ### 加入到 matrix 中 。
    #         nums=np.zeros((12), dtype='float')
    #         nums[len-1]=1.0
    #         matrix = np.array(nums)
    #         first_ele = False
    #     else:
    #         nums = np.zeros((12), dtype='float')
    #         nums[len - 1] = 1.0
    #         matrix = np.c_[matrix, nums]
    # matrix = matrix.transpose()

    """
    a = []
    for x in range(0, 5610):
        result = [float(item) for item in matrix[x]]
        a.append(result)
    arr = np.array(a)
    """
    f.close()
    #print(matrix)
    #print(type(matrix))
    #print(matrix.dtype)
    #print(matrix.shape)
    return matrix


train_labels_path = readFileInt(r'D:\学习笔记\ll_code\Cost最新修改\最新修改——姿势识别fedavg修改节省通讯成本3个取最大 - 副本\data\Train\y_train.txt')
# print(train_labels_path)
test_labels_path = readFileInt(r'D:\学习笔记\ll_code\Cost最新修改\最新修改——姿势识别fedavg修改节省通讯成本3个取最大 - 副本\data\Test\y_test.txt')
for i in range(1233):
    train_labels_path.append(test_labels_path[i])
print(train_labels_path)
print(len(train_labels_path))

index1 = []
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 1:
        index1.append(i)
        a = a+1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 2:
        index1.append(i)
        a = a + 1
print(a)
a = 0

for i in range(len(train_labels_path)):
    if train_labels_path[i] == 3:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 4:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 5:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 6:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 7:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 8:
        index1.append(i)
        a = a+1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 9:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 10:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 11:
        index1.append(i)
        a = a + 1
print(a)
a = 0
for i in range(len(train_labels_path)):
    if train_labels_path[i] == 12:
        index1.append(i)
        a = a + 1
print(a)
a = 0
print(index1)
print(len((index1)))
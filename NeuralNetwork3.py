import sys
import numpy as np
from math import exp, log
from time import time


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/np.sum(e_x,axis=0)


def sigmoid_v(x):
    return 1/(1+np.exp(-x))


def d_sigmoid_v(x):
    return x*(1-x)


def delta(i,j):
    if i == j:
        return 1
    else:
        return 0


def ReLU(x):
    return x * (x > 0)


def d_ReLU(x):
    return np.heaviside(x, 0)

t_start = time()
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))


file_name1 = sys.argv[1]
file_name2 = sys.argv[2]
file_name3 = sys.argv[3]

train_img_file = open(file_name1,'r')
train_label_file = open(file_name2,'r')
test_img_file = open(file_name3,'r')

n1 = 256 # number of nodes in hidden layer 1
n2 = 128 # number of nodes in hidden layer 2

learning_rate = 0.5


batch_size = 10
learning_rate = learning_rate/batch_size
epoch_size =10
input = np.zeros((784,1))
h1 = np.zeros((n1,1))
h2 = np.zeros((n2,1))
b1 = np.random.randn(n1,1)
b2 = np.random.randn(n2,1)
b3 = np.random.randn(10,1)

h1_d = np.zeros((n1,1))
h2_d = np.zeros((n2,1))
O = np.zeros((10,1))
dL_dPO = np.zeros((10,1))
dO_dPO = np.zeros((10,10))

W_I_1 = np.random.randn(n1,784)*np.sqrt(1./784)
W_1_2 = np.random.randn(n2,n1)*np.sqrt(1./n1)
W_2_P = np.random.randn(10,n2)*np.sqrt(1./n2)

temp_d_W_I_1 = np.zeros((n1,784))
temp_d_W_1_2 = np.zeros((n2,n1))
temp_d_W_2_P = np.zeros((10,n2))
temp_d_b1 = np.zeros((n1, 1))
temp_d_b2 = np.zeros((n2, 1))
temp_d_b3 = np.zeros((n2, 1))

cuml_d_W_I_1 = np.zeros((n1,784))
cuml_d_W_1_2 = np.zeros((n2,n1))
cuml_d_W_2_P = np.zeros((10,n2))
cuml_d_b1 = np.zeros((n1, 1))
cuml_d_b2 = np.zeros((n2, 1))
cuml_d_b3 = np.zeros((10, 1))

train_img_data = []
train_label_data= []

line = train_img_file.readline()
while line:
    data = np.array([int(w) for w in line.strip().split(',')]).reshape(784,1)
    train_img_data.append(data)
    probability = np.zeros(10)
    probability[int(train_label_file.readline().strip())] = 1
    probability = np.array(probability).reshape(10,1)
    train_label_data.append(probability)
    line = train_img_file.readline()

train_label_data = np.array(train_label_data)
train_img_data = np.array(train_img_data)/255
print(train_label_data.shape)
print(train_img_data.shape)

train_img_file.close()
train_label_file.close()

pick = np.arange(60000)
np.random.shuffle(pick)
sequence = pick[:59999]
# sequence = np.arange(60000)

for epoch in range(epoch_size):
    if(time()-t_start>1680.0):
        break
    print("Processing epoch ",epoch)
    np.random.shuffle(sequence)
    crt_count = 0
    for img in range(len(sequence)):
        # print("img: ",img)
        I = train_img_data[sequence[img]]
        # print(I)
        target = train_label_data[sequence[img]]
        # Forward
        if img%batch_size == 0:
            #print("Updating parameters!")
            #print(cuml_d_W_I_1)
            #update matrices
            W_I_1 -= learning_rate*cuml_d_W_I_1
            W_1_2 -= learning_rate*cuml_d_W_1_2
            W_2_P -= learning_rate*cuml_d_W_2_P
            b1 -= learning_rate*cuml_d_b1
            b2 -= learning_rate*cuml_d_b2
            b3 -= learning_rate*cuml_d_b3
            cuml_d_W_I_1 = np.zeros((n1, 784))
            cuml_d_W_1_2 = np.zeros((n2, n1))
            cuml_d_W_2_P = np.zeros((10, n2))
            cuml_d_b1 = np.zeros((n1,1))
            cuml_d_b2 = np.zeros((n2,1))
            cuml_d_b3 = np.zeros((10,1))
            #print("Image number ",img)

        h1 = np.dot(W_I_1,I)+b1
        h1 = sigmoid_v(h1)
        # h1 = ReLU(h1)
        h2 = np.dot(W_1_2,h1)+b2
        h2 = sigmoid_v(h2)
        # h2 = ReLU(h2)
        PO = np.dot(W_2_P,h2)+b3
        O = softmax(PO)
        h2_d = d_sigmoid_v(h2)
        # h2_d = d_ReLU(h2)
        h1_d = d_sigmoid_v(h1)
        # h1_d = d_ReLU(h1)

        if list(O).index(max(list(O))) == list(target).index(1):
            crt_count += 1

        # print(PO)
        # print(O)
        # print(target)
        dL_dPO = O - target
        # print(target)
        # print(dL_dO)

        # for m in range(10):
        #     for p in range(10):
        #         dO_dPO[m][p] = O[m]*(delta(m,p)-O[p])
        # print(dO_dPO)
        # exit()

        delta_3 = dL_dPO
        delta_2 = np.multiply(np.dot(W_2_P.transpose(), delta_3),h2_d)
        delta_1 = np.multiply(np.dot(W_1_2.transpose(), delta_2),h1_d)

        ###########
        # Value for W_2_P:
        # for p in range(10):
        #     for k in range(n2):
        #         sum = 0
        #         for m in range(10):
        #             sum += dL_dO[m]*O[m]*(delta(m,p)-O[p])*h2[k]
        #         temp_d_W_2_P[p][k] = sum
        temp_d_W_2_P = np.dot(delta_3,h2.transpose())
        temp_d_b3 = np.sum(delta_3, axis = 1, keepdims=True)
        ##############
        # Value for W_1_2:
        # for k in range(n2):
        #     for j in range(n1):
        #         sum = 0
        #         for m in range(10):
        #             subsum = 0
        #             for n in range(10):
        #                 subsum += O[m]*(delta(m,n)-O[n])*W_2_P[n][k]*h2[k]*(1-h2[k])*h1[j]
        #             sum += dL_dO[m]*subsum
        #         temp_d_W_1_2[k][j] = sum
        temp_d_W_1_2 = np.dot(delta_2,h1.transpose())
        temp_d_b2 = np.sum(delta_2, axis=1, keepdims=True)
        ###############
        # Value for W_I_1:
        # for j in range(n1):
        #     for i in range(784):
        #         sum = 0
        #         for m in range(10):
        #             subsum = 0
        #             for n in range(10):
        #                 subsubsum = 0
        #                 for w in range(n2):
        #                     subsubsum += W_2_P[n][w]*h2[w]*(1-h2[w])*W_1_2[w][j]*h1[j]*(1-h1[j])*I[i]
        #                 subsum += O[m]*(delta(m,n)-O[n])*subsubsum
        #             sum += dL_dO[m]*subsum
        #         temp_d_W_I_1[j][i] = sum
        # print(delta_1)
        temp_d_W_I_1 = np.dot(delta_1, I.transpose())
        temp_d_b1 = np.sum(delta_1, axis=1, keepdims=True)
        # print(temp_d_W_I_1)
        ###############

        cuml_d_W_I_1 += temp_d_W_I_1
        cuml_d_W_1_2 += temp_d_W_1_2
        cuml_d_W_2_P += temp_d_W_2_P
        cuml_d_b1 += temp_d_b1
        cuml_d_b2 += temp_d_b2
        cuml_d_b3 += temp_d_b3



    print("Epoch {} accuray {}".format(epoch,crt_count/len(sequence)))
    print("Time: ",time()-t_start)


test_predict_label = open('test_predictions.csv','w')
line = test_img_file.readline()
while line:
    I = np.array([int(w) for w in line.strip().split(',')]).reshape(784,1)
    I = I/255
    h1 = np.dot(W_I_1, I) + b1
    h1 = sigmoid_v(h1)
    # h1 = ReLU(h1)
    h2 = np.dot(W_1_2, h1) + b2
    h2 = sigmoid_v(h2)
    # h2 = ReLU(h2)
    PO = np.dot(W_2_P, h2) + b3
    O = softmax(PO)
    test_predict_label.write(str(list(O).index(max(list(O))))+'\n')
    line = test_img_file.readline()

test_img_file.close()

test_predict_label.close()

# calculate accuracy

test_predict_label = open('test_predictions.csv','r')
test_label = open('test_label.csv','r')

crt_count = 0
line = test_label.readline()
while line:
    try:
        a = int(line.strip())
        b = int(test_predict_label.readline().strip())
    except ValueError:
        break
    if a == b:
        crt_count += 1
    line = test_label.readline()

print("Test accuracy: ",crt_count/10000)
print("Total time: ",time()-t_start)



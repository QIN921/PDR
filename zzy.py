import matplotlib.pyplot as plt
import numpy as np
# import xlrd  # 导入库
# 打开文件
import csv

import numpy as np

def get_xuanzhuan(gyroscope,t):
    theta_speed = np.sqrt(np.sum(gyroscope ** 2))
    n = gyroscope / theta_speed
    print(n)
    theta = theta_speed*t
    theta = theta/180*np.pi
    cos = np.cos(theta)
    sin =np.sin(theta)
    sin=-sin
    [nx,ny,nz]=n
    T = np.zeros(shape=(3,3))
    T[0][0] = nx**2*(1-cos)+cos
    T[0][1] = nx*ny*(1-cos)-nz*sin
    T[0][2] = nx*nz*(1-cos)+ny*sin
    T[1][0] = nx*ny*(1-cos)+nz*sin
    T[1][1] = ny**2*(1-cos)+cos
    T[1][2] = ny*nz*(1-cos)-nx*sin
    T[2][0] = nx*nz*(1-cos)-ny*sin
    T[2][1] = ny*nz*(1-cos)+nx*sin
    T[2][2] = nz**2*(1-cos)+cos
    T_invovle = np.linalg.inv(T)
    return T,T_invovle

filename = './position.csv'
data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # header = next(csv_reader)        # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        data.append([row[2],row[3],row[9]])  # 选择某一列加入到data数组中
    # print(data)
def get_z(index):
    filename2 = './running.csv'
    data2=[]
    with open(filename2) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            data2.append([row[8],row[9],row[10],row[11],row[12],row[13],row[17],row[20],row[14]])  # 选择某一列加入到data数组
    acc = []
    gros = []
    t = []
    t_delta=[]
    begin = 0
    for i in data2:
        if i[7] == index: #or i[2]=='28' or i[2]=='29':
            if eval(i[8])==1 and begin == 0:
                continue
            begin = 1
            acc.append([eval(i[0]),eval(i[1]),eval(i[2])])
            gros.append([eval(i[3]),eval(i[4]),eval(i[5])])
            t.append(eval(i[6]))
    acc = np.array(acc)
    gros = np.array(gros)
    t = np.array(t)
    t = t/1000
    length = len(t)-1
    T = []
    T_i = []
    T1_linshi = np.zeros(shape=(3, 3))
    T1_linshi[0][0] = 1
    T1_linshi[1][1] = 1
    T1_linshi[2][2] = 1
    T2_linshi = np.zeros(shape=(3, 3))
    T2_linshi[0][0] = 1
    T2_linshi[1][1] = 1
    T2_linshi[2][2] = 1
    for i in range(length):
        t_delta.append(t[i+1]-t[i])
        T1,T2 = get_xuanzhuan(gros[i], t_delta[i])
        if index == '30' or '31' or '32':
            acc[i] = np.dot(T2_linshi, acc[i])
        T1_linshi = np.dot(T1_linshi, T1)
        T2_linshi = np.dot(T2, T2_linshi)
        T.append(T1_linshi)
        T_i.append(T2_linshi)
    return t,acc
# #01.jpg在该代码的同一路径下，所以没有引用路径
# img = plt.imread('C:/Users/zzzzzzzzz/Desktop/学习/软件课设/软件课程设计2022资料/室内地图/1.jpg',5)
# #plt.style.use('dark_background')#画布是黑色背景
# fig,ax = plt.subplots(figsize=(7,7),dpi=200)
# ax.imshow(img,extent=[-7.1+1.65,7.1+1.65,-5.2,6])
#fig.patch.set_alpha(1) #设置透明度
t,acc = get_z('30')
#ax.plot(x,y,color='r',linestyle='-.')
acc_z = []
acc_x = []
acc_y = []
for i in range(len(t)):
    acc_z.append(acc[i][2])
    acc_x.append(acc[i][0])
    acc_y.append(acc[i][1])

# ax.plot(t,acc_z,color='r')
# ax.scatter(x,y,color='k',marker='o')
# 设置字体格式
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}


# 选取画布，figsize控制画布大小
fig = plt.figure(figsize=(7,5))
# fig = plt.figure()

# 绘制子图 1,2,1 代表绘制 1x2 个子图，本图为第 1 个，即 121
# ax 为本子图
ax = fig.add_subplot(3, 1, 1)
# 绘图 # 具体线型、颜色、label可搜索该函数参数
ax.plot(t,acc_z,color='r' )
# ax 子图的 x,y 坐标 label 设置
# 使用 r'...$\gamma$' 的形式可引入数学符号
# font 为label字体设置
ax.set_xlabel(r'timestamp', font1)
ax.set_ylabel("g", font1)
# 坐标轴字体设置
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# 添加子图 2，具体方法与图 1 差别不大
ax2 = fig.add_subplot(3, 1, 2)
# plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
ax2.plot(t,acc_x,color='red' )  #o-:圆形
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax3 = fig.add_subplot(3, 1, 3)
# plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
ax3.plot(t,acc_y,color='blue' )  #o-:圆形
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]


# 选取画布，figsize控制画布大小
# fig2 = plt.figure(figsize=(7,5))
# fig = plt.figure()

# 绘制子图 1,2,1 代表绘制 1x2 个子图，本图为第 1 个，即 121
# ax 为本子图
# ax4 = fig2.add_subplot(3, 1, 1)
# # 绘图 # 具体线型、颜色、label可搜索该函数参数
# ax4.plot(x1, y5,color='r' )
# # ax 子图的 x,y 坐标 label 设置
# # 使用 r'...$\gamma$' 的形式可引入数学符号
# # font 为label字体设置
# ax4.set_xlabel(r'timestamp', font1)
# ax4.set_ylabel("g", font1)
# # 坐标轴字体设置
# labels = ax4.get_xticklabels() + ax4.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # 添加子图 2，具体方法与图 1 差别不大
# ax5 = fig2.add_subplot(3, 1, 2)
# # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
# ax5.plot(x1, y5,color='red' )  #o-:圆形
# labels = ax5.get_xticklabels() + ax5.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# ax6 = fig2.add_subplot(3, 1, 3)
# # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
# ax6.plot(x1, y6,color='blue' )  #o-:圆形
# labels = ax6.get_xticklabels() + ax6.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

# 保存及展示
#plt.savefig('hyperparams.eps')
plt.show()

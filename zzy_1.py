import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import csv
import math
def get_xuanzhuan(gyroscope,t):
    gyroscope = np.array(gyroscope)
    theta_speed = np.sqrt(np.sum(gyroscope ** 2))
    n = gyroscope / theta_speed
    # print(n)
    theta = theta_speed*t
    theta = theta/180*np.pi
    cos = np.cos(theta)
    sin =np.sin(theta)
    sin=sin
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
def __find_peak__(acc):
    p_data = np.zeros_like(acc, dtype=np.int32)
    count = len(acc)
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if acc[i] > acc[i - k] and acc[i] > acc[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    print(max_window_length)
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if acc[i] > acc[i - k] and acc[i] > acc[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]
class PDR:
    def __init__(self,csv_postion,csv_running,sample_batch):
        self.sample_batch = sample_batch
        self.accx = []
        self.accy = []
        self.accz = []
        self.gyroscopex = []
        self.gyroscopey = []
        self.gyroscopez = []
        self.stay = []
        self.timestamp = []
        self.step_length = 0
        self.step_frequency = []
        self.x_position = []
        self.y_position = []
        self.T = []
        self.T_i = []
        self.K = 0.7
        self.x2_position = []
        self.y2_position = []
        self.model = 0
        if sample_batch in ['30','31','32']:
            self.model = 1
        with open(csv_postion) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                if row[9] == sample_batch:
                    self.x_position.append(eval(row[2]))
                    self.y_position.append(eval(row[3]))

        with open(csv_running) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            begin = 0
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                if row[20] == sample_batch:
                    self.stay.append(eval(row[14]))
                    if eval(row[14]) == 1 and begin == 0:
                        continue
                    begin = 1
                    self.accx.append(eval(row[8]))
                    self.accy.append(eval(row[9]))
                    self.accz.append(eval(row[10]))
                    self.gyroscopex.append(eval(row[11])/1300*90)
                    self.gyroscopey.append(eval(row[12])/1300*90)
                    self.gyroscopez.append(eval(row[13])/1300*90)
                    self.timestamp.append(eval(row[17]))
        a = self.timestamp[0]
        for i in range(len(self.timestamp)):
            self.timestamp[i] -= a
            self.timestamp[i] = self.timestamp[i]/1000
        length = len(self.timestamp) - 1
        T1_linshi = np.zeros(shape=(3, 3))
        T1_linshi[0][0] = 1
        T1_linshi[1][1] = 1
        T1_linshi[2][2] = 1
        T2_linshi = np.zeros(shape=(3, 3))
        T2_linshi[0][0] = 1
        T2_linshi[1][1] = 1
        T2_linshi[2][2] = 1
        for i in range(length):
            T1, T2 = get_xuanzhuan([self.gyroscopex[i],self.gyroscopey[i],self.gyroscopez[i]], self.timestamp[i+1]-self.timestamp[i])
            [self.accx[i], self.accy[i], self.accz[i]] = np.dot(T1_linshi, [self.accx[i],self.accy[i],self.accz[i]])
            T1_linshi = np.dot(T1_linshi,T1)
            T2_linshi = np.dot(T2, T2_linshi)
            self.T.append(T1_linshi)
            self.T_i.append(T2_linshi)
        print(self.T)
        self.accx = np.array(self.accx)
        self.accy = np.array(self.accy)
        self.accz = np.array(self.accz)
        self.gyroscopex = np.array(self.gyroscopex)
        self.gyroscopey = np.array(self.gyroscopey)
        self.gyroscopez = np.array(self.gyroscopez)
    def vis(self):
        L = []
        y = []
        if self.sample_batch in ['30','31','32']:

            p_max = __find_peak__(self.accy)
            p_min = __find_peak__(-1*self.accy)
            plt.plot(self.timestamp, self.accy)
            for i in p_max:
                L.append(self.timestamp[i])
                y.append(self.accy[i])
            min_len = min(len(p_max),len(p_min))
            a_max = 0
            a_min = 0
            for i in range(min_len):
                a_max += self.accy[p_max[i]]
                a_min += self.accy[p_min[i]]
            a_max /= min_len
            a_min /= min_len
            delta = (a_max - a_min) / 16384 * 9.8
            self.step_length  = self.K * math.pow(delta,0.25)
        else:
            p_max = __find_peak__(self.accz)
            p_min = __find_peak__(-1*self.accz)
            plt.plot(self.timestamp, self.accz)
            for i in p_max:
                L.append(self.timestamp[i])
                y.append(self.accz[i])
            min_len = min(len(p_max), len(p_min))
            a_max = 0
            a_min = 0
            for i in range(min_len):
                a_max += self.accz[p_max[i]]
                a_min += self.accz[p_min[i]]
            a_max /= min_len
            a_min /= min_len
            delta = (a_max - a_min) / 16384 * 9.8
            self.step_length  = self.K * math.pow(delta,0.25)
        self.x2_position = []
        self.y2_position = []
        x_tmp = self.x_position[0]
        y_tmp = self.y_position[0]
        self.x2_position.append(x_tmp)
        self.y2_position.append(y_tmp)
        if self.model == 0:
            for i in p_max:
                E0 = -1*self.T[i][0][0]
                E3 = -1*self.T[i][1][0]
                nx = E0/np.sqrt(E0**2+E3**2)
                ny = E3/np.sqrt(E0**2+E3**2)
                x_tmp = x_tmp + ny*self.step_length
                y_tmp = y_tmp + nx*self.step_length
                self.x2_position.append(x_tmp)
                self.y2_position.append(y_tmp)
        else:
            for i in p_max:
                E0 = 1*self.T[i][0][0]
                E6 = 1*self.T[i][2][0]
                nx = E0/np.sqrt(E0**2+E6**2)
                ny = E6/np.sqrt(E0**2+E6**2)
                x_tmp = x_tmp + ny*self.step_length
                y_tmp = y_tmp - nx*self.step_length
                self.x2_position.append(x_tmp)
                self.y2_position.append(y_tmp)

        print(self.x2_position)
        print(self.y2_position)
        #因为第一种情况
        print(len(L))
        print(L)
        plt.scatter(L, y, color="red")
        plt.show()
    def plot_acc(self):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,}

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(3, 1, 1)
        # 绘图 # 具体线型、颜色、label可搜索该函数参数
        ax.plot(P1.timestamp, P1.accx, color='r')
        ax.set_xlabel(r'timestamp', font1)
        ax.set_ylabel("g", font1)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # 添加子图 2，具体方法与图 1 差别不大
        ax2 = fig.add_subplot(3, 1, 2)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax2.plot(P1.timestamp, P1.accy, color='red')  # o-:圆形
        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax3 = fig.add_subplot(3, 1, 3)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax3.plot(P1.timestamp, P1.accz, color='blue')  # o-:圆形
        labels = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.show()
    def plot_gro(self):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,}
        # 选取画布，figsize控制画布大小
        fig2 = plt.figure(figsize=(7, 5))
        # fig = plt.figure()

        # 绘制子图 1,2,1 代表绘制 1x2 个子图，本图为第 1 个，即 121
        # ax 为本子图
        ax4 = fig2.add_subplot(3, 1, 1)
        # 绘图 # 具体线型、颜色、label可搜索该函数参数
        ax4.plot(P1.timestamp, P1.gyroscopex, color='r')
        # ax 子图的 x,y 坐标 label 设置
        # 使用 r'...$\gamma$' 的形式可引入数学符号
        # font 为label字体设置
        ax4.set_xlabel(r'timestamp', font1)
        ax4.set_ylabel("g", font1)
        # 坐标轴字体设置
        labels = ax4.get_xticklabels() + ax4.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # 添加子图 2，具体方法与图 1 差别不大
        ax5 = fig2.add_subplot(3, 1, 2)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax5.plot(P1.timestamp, P1.gyroscopey, color='red')  # o-:圆形
        labels = ax5.get_xticklabels() + ax5.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax6 = fig2.add_subplot(3, 1, 3)
        # plt.plot(x,k1,'s-',color = 'r',label="红线的名字")#s-:方形
        ax6.plot(P1.timestamp, P1.gyroscopez, color='blue')  # o-:圆形
        labels = ax6.get_xticklabels() + ax6.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # 保存及展示
        # plt.savefig('hyperparams.eps')
        plt.show()

    def z_inter(self):
        jifen = []
        tmp = 0
        for i in range(len(self.timestamp)):
            tmp = tmp +self.gyroscopex[i]*self.timestamp[i]
            jifen.append(tmp)
        plt.scatter(self.timestamp, jifen, color="red")
        print(jifen)
        plt.show()
    # def __Weinberg__():

    # def find_direction(self):


csv_postion = './position.csv'
csv_running = './running.csv'
sample_batch = '28'
P1 = PDR(csv_postion,csv_running,sample_batch)
P1.z_inter()
P1.vis()
P1.plot_acc()
P1.plot_gro()
print(P1.step_length)

# P1.z_jifen()
img = plt.imread('./background.png',5)
fig,ax = plt.subplots(figsize=(7,7),dpi=200)
ax.imshow(img,extent=[-7.1+1.65,7.1+1.65,-5.2,6])
ax.plot(P1.x_position,P1.y_position,color='r')
plt.show()

img = plt.imread('./background.png',5)
fig,ax = plt.subplots(figsize=(7,7),dpi=200)
ax.imshow(img,extent=[-7.1+1.65,7.1+1.65,-5.2,6])
ax.plot(P1.x2_position,P1.y2_position,color='r')
plt.show()



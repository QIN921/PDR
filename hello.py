import csv
import matplotlib.pyplot as plt
import numpy as np
import pywt
import math
import position


class Running:
    def __init__(self):
        # self.cor = []
        # self.inv_cor = []

        self.sample_batch = 0
        self.accx = []
        self.accy = []
        self.accz = []
        self.gyroscopex = []
        self.gyroscopey = []
        self.gyroscopez = []
        self.timestamp = []
        self.model = 1  # 记录运动模式,0为static,1为dynamic
        # 以上信息通过读入，并通过筛选删除与修改
        # 迈步的z轴加速度最大与最小
        self.step_max = []
        self.step_min = []
        # 迈步时刻
        self.step_time = []
        # 步数
        self.step = 0
        # 每一步的长度
        self.length = []
        # 每一步的方向
        self.angle = []
        # 每个定位点的位置
        self.position_x = []
        self.position_y = []
        # 每个定位点的误差
        self.error = []

    def plot_acc(self):
        plt.figure()  # 初始化一张图
        x = self.timestamp
        plt.subplot(3, 1, 1)
        plt.plot(x, self.accx)  # 连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
        plt.subplot(3, 1, 2)
        plt.plot(x, self.accy)
        plt.subplot(3, 1, 3)
        # if self.step_max:
        #     x_max = [i[0] for i in self.step_max]
        #     y_max = [i[1] for i in self.step_max]
        #     plt.scatter(x_max, y_max, c='r')
        # if self.step_min:
        #     x_min = [i[0] for i in self.step_min]
        #     y_min = [i[1] for i in self.step_min]
        #     plt.scatter(x_min, y_min, c='g')
        plt.plot(x, self.accz)
        # plt.show()

    def plot_gyroscope(self):
        plt.figure()  # 初始化一张图
        x = self.timestamp
        plt.subplot(3, 1, 1)
        plt.plot(x, self.gyroscopex)  # 连线图,若要散点图将此句改为：plt.scatter(x,y) #散点图
        plt.subplot(3, 1, 2)
        plt.plot(x, self.gyroscopey)
        plt.subplot(3, 1, 3)
        # threshold_1 = [-150 for _ in range(len(x))]
        # threshold_2 = [-700 for _ in range(len(x))]
        # plt.plot(x, threshold_1, 'r')
        # plt.plot(x, threshold_2, 'g')
        plt.plot(x, self.gyroscopez, marker='x')
        # 以下为动态时计步显示
        if self.model == 1:
            if self.step_max:
                x_max = [i[0] for i in self.step_max]
                y_max = [i[1] for i in self.step_max]
                plt.scatter(x_max, y_max, c='r')
            if self.step_min:
                x_min = [i[0] for i in self.step_min]
                y_min = [i[1] for i in self.step_min]
                plt.scatter(x_min, y_min, c='g')
        # plt.show()

    def step_counter_dynamic(self):
        step = 0
        gyroscopez = self.gyroscopez
        mx, mn = 0, 0
        lens = len(gyroscopez)
        i = 0
        mx_index, mn_index = 0, 0
        while i < lens:
            if gyroscopez[i] > 500:
                # 开始寻找极大值，并将极小值存起来，即步数加一
                if mn < -500:
                    step += 1
                    mn = 0
                    self.step_max.append([self.timestamp[mn_index], self.gyroscopez[mn_index]])
                if gyroscopez[i] > mx:
                    mx = gyroscopez[i]
                    mx_index = i
            elif gyroscopez[i] < -500:
                if mx > 500:
                    step += 1
                    mx = 0
                    self.step_min.append([self.timestamp[mx_index], self.gyroscopez[mx_index]])
                if gyroscopez[i] < mn:
                    mn = gyroscopez[i]
                    mn_index = i
            i += 1

        step_time = []
        for index in range(len(self.step_max)):
            step_time.append([self.step_min[index][0], self.step_max[index][0]])
        self.step_time = step_time

        # if self.step_time[-1][1] > 0 and mn_index != lens - 1:
        #     step += 1
        #     self.step_time.append([self.timestamp[mn_index], self.gyroscopez[mn_index]])
        # elif self.step_time[-1][1] < 0 and mx_index != lens - 1:
        #     step += 1
        #     self.step_time.append([self.timestamp[mx_index], self.gyroscopez[mx_index]])
        self.step = step

    def step_counter_static(self):
        step = 0
        threshold_value = 0.05
        threshold_time = 50
        accz = self.accz
        timestamp = self.timestamp
        mx, mn = [], []
        pre_index = 0

        for index, value in enumerate(accz):
            if index == 0:
                continue
            if value > 0 and (value - accz[index - 1]) > threshold_value \
                    and (timestamp[index] - timestamp[index - 1]) > threshold_time:
                while index < len(accz) - 1 and accz[index] < accz[index + 1]:
                    index += 1

                # 在波峰的前temp个值中找波谷
                temp = 4
                if temp < index:
                    l = accz[index - temp: index]
                    # i = accz.index(min(l))
                    accz = np.array(accz)
                    i = np.where(accz == min(l))[0][0]
                    if pre_index >= i or [timestamp[i], accz[i]] in mn:
                        last = mx.pop()
                        if accz[index] > last[1]:
                            mx.append([timestamp[index], accz[index]])
                            pre_index = index
                        else:
                            mx.append(last)
                    else:
                        step += 1
                        pre_index = index
                        mx.append([timestamp[index], accz[index]])
                        mn.append([timestamp[i], accz[i]])

        # print("原有数据：" + str(len(accz)) + "条")
        # print("检测到波峰：" + str(len(mx)) + '个')
        # print("检测到波谷：" + str(len(mn)) + '个')
        self.step_max = mx
        self.step_min = mn
        self.step = step
        step_time = []
        for index in range(len(mx)):
            step_time.append([mn[index][0], mx[index][0]])
        self.step_time = step_time

    def step_length(self):
        length = []
        K = 0.5
        g = 9.8
        l = len(self.step_max)
        for index in range(l):
            res = K * (self.step_max[index][1] * g - self.step_min[index][1] * g) ** 0.25
            length.append(res)
        self.length = length

    def cal_angle(self, offset):
        [q0, q1, q2, q3] = [1, 0, 0, 0]
        gamma, theta, phi = [], [], []
        ax = sum(self.accx) / len(self.accx)
        ay = sum(self.accy) / len(self.accy)
        az = sum(self.accz) / len(self.accz)
        # T = np.zeros(shape=(3, 3))
        length = len(self.timestamp)
        alpha = offset
        for i in range(length - 1):
            dt = (self.timestamp[i + 1] - self.timestamp[i]) / 1000
            [gx, gy, gz] = [self.gyroscopex[i] / 1250 * 90 * np.pi / 180, self.gyroscopey[i] / 1250 * 90 * np.pi / 180,
                            self.gyroscopez[i] / 1250 * 90 * np.pi / 180]

            # T[0][0] = 1 - 2 * (q2 * q2 + q3 * q3)
            # T[0][1] = 2 * (q1 * q2 + q0 * q3)
            # T[0][2] = 2 * (q1 * q3 - q0 * q2)
            # T[1][0] = 2 * (q1 * q2 - q0 * q3)
            # T[1][1] = 1 - 2 * (q1 * q1 + q3 * q3)
            # T[1][2] = 2 * (q2 * q3 + q0 * q1)
            # T[2][0] = 2 * (q1 * q3 + q0 * q2)
            # T[2][1] = 2 * (q2 * q3 - q0 * q1)
            # T[2][2] = 1 - 2 * (q1 * q1 + q2 * q2)
            ax, ay, az = self.accx[i], self.accy[i], self.accz[i]
            alpha += (ax * self.gyroscopex[i] / 1250 * 90 + ay * self.gyroscopey[i] / 1250 * 90
                      + az * self.gyroscopez[i] / 1250 * 90)*dt / np.sqrt(ax ** 2 + ay ** 2 + az ** 2) * 2
            # sign = np.sign(alpha)
            # if abs(abs(alpha) - 90) < 20:
            #     alpha = 90 * sign
            # elif abs(abs(alpha) - 180) < 20:
            #     alpha = 180 * sign
            # elif abs(abs(alpha) - 270) < 20:
            #     alpha = 270 * sign
            # elif abs(abs(alpha) - 360) < 20:
            #     alpha = 360 * sign
            # elif abs(abs(alpha) - 0) < 5:
            #     alpha = 0
            theta.append(alpha)
            gx = gx * 0.5 * dt
            gy = gy * 0.5 * dt
            gz = gz * 0.5 * dt
            q0 = q0 - q1 * gx - q2 * gy - q3 * gz
            q1 = q1 + q0 * gx + q2 * gz - q3 * gy
            q2 = q2 + q0 * gy - q1 * gz + q3 * gx
            q3 = q3 + q0 * gz + q1 * gy - q2 * gx
            [q0, q1, q2, q3] = Vsqrt([q0, q1, q2, q3])
            g1 = 2 * (q1 * q3 - q0 * q2)
            g2 = 2 * (q2 * q3 + q0 * q1)
            g3 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
            g4 = 2 * (q1 * q2 + q0 * q3)
            g5 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
            gamma.append(- np.arcsin(g1) * 180 / np.pi)
            # theta.append(- np.arctan2(g2, g3) * 180 / np.pi * 2 + offset)
            phi.append(- np.arctan2(g4, g5) * 180 / np.pi + offset)

        self.timestamp = self.timestamp[:-1]
        theta = denoise(theta)
        if self.model == 0:
            self.angle = phi
        elif self.model == 1:
            self.angle = theta

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.scatter(self.timestamp, gamma)
        plt.subplot(3, 1, 2)
        plt.scatter(self.timestamp, theta)
        time = [i for r in self.step_time for i in r]
        for i in range(len(self.timestamp)):
            if self.timestamp[i] in time:
                plt.scatter(self.timestamp[i], theta[i], c='r')
        plt.subplot(3, 1, 3)
        plt.scatter(self.timestamp, phi)

    def invalid_time(self):
        invalid = []
        lens = len(self.gyroscopex)
        mean_x = abs(sum(self.gyroscopex) / lens) + 10
        mean_y = abs(sum(self.gyroscopey) / lens) + 10
        i = 0
        while i < lens - 1:
            if abs(self.gyroscopex[i]) > 200:
                # 此时运动状态不是水平移动
                offset_left = 1
                while i > offset_left and abs(self.gyroscopex[i - offset_left]) >= mean_x:
                    offset_left += 1
                offset_right = 1
                while i + offset_right < lens - 2 and abs(self.gyroscopex[i + offset_right]) >= mean_x:
                    offset_right += 1
                if i == lens - 1:
                    invalid.append([i - offset_left, lens - 1])
                else:
                    invalid.append([i - offset_left, i + offset_right])
                i += offset_right
            elif abs(self.gyroscopey[i]) > 150:
                # 此时运动状态不是水平移动
                offset_left = 1
                while i > offset_left and abs(self.gyroscopey[i - offset_left]) >= mean_y:
                    offset_left += 1
                offset_right = 1
                while i + offset_right < lens - 1 and abs(self.gyroscopey[i + offset_right]) >= mean_y:
                    offset_right += 1
                if i == lens - 1:
                    invalid.append([i - offset_left, lens - 1])
                else:
                    invalid.append([i - offset_left, i + offset_right])
                i += offset_right
            else:
                i += 1
        # print(invalid)
        for i in invalid:
            self.accx = np.delete(self.accx, np.s_[i[0]:i[1] + 1], 0)
            self.accy = np.delete(self.accy, np.s_[i[0]:i[1] + 1], 0)
            self.accz = np.delete(self.accz, np.s_[i[0]:i[1] + 1], 0)
            self.gyroscopex = np.delete(self.gyroscopex, np.s_[i[0]:i[1] + 1], 0)
            self.gyroscopey = np.delete(self.gyroscopey, np.s_[i[0]:i[1] + 1], 0)
            self.gyroscopez = np.delete(self.gyroscopez, np.s_[i[0]:i[1] + 1], 0)
            self.timestamp = np.delete(self.timestamp, np.s_[i[0]:i[1] + 1], 0)

    def pdr_position(self, init_position=(0, 0)):
        # print(len(self.timestamp))
        # print(self.timestamp)
        # print(len(self.length))
        # print(len(self.step_time))
        # print(len(self.angle))
        position_x = [init_position[0]]
        position_y = [init_position[1]]
        x = init_position[0]
        y = init_position[1]
        counter = 0
        if self.model == 1:
            step_time = [i for arr in self.step_time for i in arr]
        elif self.model == 0:
            step_time = [i[1] for i in self.step_time]
        for i in range(len(self.timestamp)):
            # if counter == len(self.step_time):
            #     break
            if self.timestamp[i] in step_time:
                if self.model == 0:
                    length = self.length[counter]
                else:
                    length = 0.7
                # print(self.angle[i])
                x += length * np.cos(self.angle[i] * np.pi / 180)
                y += length * np.sin(self.angle[i] * np.pi / 180)
                counter += 1
                # if x > 7.647:
                #     x = 7.6
                # elif x < -4.653:
                #     x = -4.6
                # if y < -3.906:
                #     y = -3.906
                # elif y > 4.694:
                #     y = 4.6
                position_x.append(x)
                position_y.append(y)
        self.position_x = position_x
        self.position_y = position_y


def plot_position(r: Running, p: position.Position):
    position_x = r.position_x
    position_y = r.position_y

    img = plt.imread('background.png')
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    ax.imshow(img, extent=[-4.5 - 0.153, 7.8 - 0.153, -3.4 - 0.506, 5.2 - 0.506])
    plt.scatter(0, 0, c='r', s=10)
    plt.plot(position_x, position_y, marker='1')

    x = p.x
    y = p.y

    plt.scatter(x, y, c='g', s=10, marker='p')

    gt_x = [[-1, -1], [-1, 1.5], [1.5, 1.5]]
    gt_y = [[3.4, -3.2], [-3.2, -3.2], [-3.2, 3.4]]
    for i in range(len(gt_x)):
        plt.plot(gt_x[i], gt_y[i], c='r')


def Vsqrt(l: list):
    temp = 0
    for i in l:
        temp += i * i
    recipNorm = 1 / np.sqrt(temp)
    ans = []
    for i in l:
        ans.append(i * recipNorm)
    return ans


def begin_point(p: position.Position, k):
    # 根据position.csv中在图片所确定的坐标系内的前k个点取平均值作为PDR算法的起始点
    lens = len(p.x)
    k_points = [[], []]
    size = 0
    for i in range(lens):
        if -4.653 < p.x[i] < 7.647 and -3.906 < p.y[i] < 4.694:
            k_points[0].append(p.x[i])
            k_points[1].append(p.y[i])
            size += 1
        if size >= k:
            break
    res = [sum(k_points[0]) / size, sum(k_points[1]) / size]
    # print(res)
    return res


def denoise(data):
    def sgn(num):
        if num > 0:
            return 1.0
        elif num == 0:
            return 0.0
        else:
            return -1.0

    w = pywt.Wavelet('sym8')
    [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 分解波

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    # print(lamda)
    usecoeffs = [ca3]

    # 软硬阈值折中的方法
    a = 0.2

    for k in range(length1):
        if abs(cd1[k]) >= lamda + 700:
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if abs(cd2[k]) >= lamda + 700:
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if abs(cd3[k]) >= lamda + 700:
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)

    return recoeffs[:len(data)]


def csv_position():
    min = 1e20
    dic = {}
    with open('running.csv', 'r', encoding='utf-8') as csvfile:
        # 调用csv中的DictReader函数直接获取数据为字典形式
        reader = csv.DictReader(csvfile)
        for each in reader:
            if each['stay'] == '1':
                continue
            batch = each['sample_batch']
            if batch not in dic.keys() or int(each['timestamp']) < min:
                min = int(each['timestamp'])
                # print("初始时间戳发生变化: ", min)
                r = Running()
                r.sample_batch = batch
                dic[batch] = r
            else:
                r = dic[batch]
            # 将数据中需要转换类型的数据转换类型。原本全是字符串（string）
            each['timestamp'] = (int(each['timestamp']) - min)
            r.timestamp.append(each['timestamp'])

            each['accx'] = int(each['accx']) / 16384
            each['accy'] = int(each['accy']) / 16384
            each['accz'] = int(each['accz']) / 16384
            if '26' < batch < '30':
                each['accz'] = each['accz'] + 1
                r.model = 0
            r.accx.append(each['accx'])
            r.accy.append(each['accy'])
            r.accz.append(each['accz'])

            each['gyroscopex'] = int(each['gyroscopex'])
            each['gyroscopey'] = int(each['gyroscopey'])
            each['gyroscopez'] = int(each['gyroscopez'])
            r.gyroscopex.append(each['gyroscopex'])
            r.gyroscopey.append(each['gyroscopey'])
            r.gyroscopez.append(each['gyroscopez'])
    # lst = ['27', '28', '29', '30', '31', '32']
    lst = ['30', '31', '32']
    # lst = ['27', '28', '29']
    # lst = ['27', '30']
    # lst = ['30']
    dic_position = position.csv_position()
    for i in lst:
        sample_batch = dic[i].sample_batch
        pos = dic_position[sample_batch]
        pos.x = denoise(pos.x)
        pos.y = denoise(pos.y)
        init_position = begin_point(dic_position[i], 10)
        if dic[i].model == 0:
            dic[i].gyroscopex = denoise(dic[i].gyroscopex)
            dic[i].gyroscopey = denoise(dic[i].gyroscopey)
            dic[i].gyroscopez = denoise(dic[i].gyroscopez)
            # dic[i].plot_gyroscope()
            dic[i].invalid_time()
            # dic[i].plot_gyroscope()
            dic[i].step_counter_static()
            dic[i].step_length()
            dic[i].cal_angle(-90)
            dic[i].pdr_position(init_position)
            plot_position(dic[i], pos)
            # dic[i].plot_acc()
        elif dic[i].model == 1:
            # dic[i].invalid_time()
            dic[i].step_counter_dynamic()
            dic[i].plot_gyroscope()
            dic[i].cal_angle(-90)
            # dic[i].plot_acc()
            # dic[i].plot_gyroscope()
            dic[i].pdr_position(init_position)
            plot_position(dic[i], pos)
    plt.show()


if __name__ == '__main__':
    csv_position()

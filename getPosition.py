import csv
import json


class Position:
    def __init__(self):
        self.sample_batch = 0
        self.x = []
        self.y = []
        self.z = []
        self.timestamp = []
        self.time_rel = []
        self.error = []
        self.sample_time = []
        self.inf = []

    def get_inf(self):
        for i in range(len(self.x)):
            dic = {'x': round(float(self.x[i]), 3), 'y': round(float(self.y[i]), 3),
                   'timestamp': float(self.time_rel[i]),
                   'time_rel': float(self.timestamp[i]),
                   'sample_batch': float(self.sample_batch),
                   'sample_time': self.sample_time[i]}
            j = json.dumps(dic)
            self.inf.append(j)


def getPosition(pos_csv):
    dic = {}
    with open(pos_csv, 'r', encoding='utf-8') as csvfile:
        # 调用csv中的DictReader函数直接获取数据为字典形式
        reader = csv.DictReader(csvfile)
        # 创建一个counts计数一下 看自己一共添加的数据条数
        counts = 0
        min = 1e20
        for each in reader:
            if each['sample_batch'] not in dic.keys():
                min = int(each['timestamp'])
                p = Position()
                dic[each['sample_batch']] = p
            else:
                p = dic[each['sample_batch']]
            # 将数据中需要转换类型的数据转换类型。原本全是字符串（string）
            each['x'] = float(each['x'])
            each['y'] = float(each['y'])
            each['z'] = float(each['z'])
            p.x.append(each['x'])
            p.y.append(each['y'])
            p.z.append(each['z'])
            p.sample_time.append(each['sample_time'])
            p.time_rel.append(each['timestamp'])
            each['timestamp'] = int(each['timestamp']) - min
            p.timestamp.append(each['timestamp'])
            p.sample_batch = each['sample_batch']

    ans = []
    for i in dic.values():
        i.get_inf()
        ans.append(i.inf)
    print(ans)
    return ans


if __name__ == '__main__':
    getPosition('./position.csv')





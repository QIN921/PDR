import pymongo
import csv


# 创建连接MongoDB数据库函数
def connection():
    # 连接本地MongoDB数据库服务
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    # 连接本地数据库(monogodb_name),没有时会自动创建
    mydb = myclient["mymongo"]
    # 创建集合
    mycol = mydb["PDR"]
    # 看情况是否选择清空
    # 方法一
    # x = mycol.delete_many({})
    # 方法二
    # myquery = {"name": {"$regex": "^F"}}
    # x = mycol.delete_many(myquery)
    # print(x.deleted_count, "个文档已删除")
    # 方法三
    # mycol.drop()
    return mycol


def insertToMongoDB(mycol):
    # 打开文件
    with open('position.csv', 'r', encoding='utf-8')as csvfile:
        # 调用csv中的DictReader函数直接获取数据为字典形式
        reader = csv.DictReader(csvfile)
        # 创建一个counts计数一下 看自己一共添加的数据条数
        counts = 0
        for each in reader:
            # 将数据中需要转换类型的数据转换类型。原本全是字符串（string）
            each['x'] = float(each['x'])
            each['y'] = float(each['y'])
            each['z'] = float(each['z'])
            x = mycol.insert_one(each)
            print(x)
            counts += 1
            print('成功添加了'+str(counts)+'条数据')


if __name__ == '__main__':
    col = connection()
    insertToMongoDB(col)





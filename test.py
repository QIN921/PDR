import pymongo

myclient = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = myclient['mymongo']

mycol = mydb["sites"]
mydict = {"name": "RUNOOB", "alexa": "10000", "url": "https://www.runoob.com"}
x = mycol.insert_one(mydict)
print(x)

dblist = myclient.list_database_names()
print(dblist)
if "mymongo" in dblist:
    print("数据库已存在！")
else:
    print("数据库不存在！")

collist = mydb.list_collection_names()
print(collist)
if "sites" in collist:
    print("集合已存在！")
else:
    print("集合不存在！")


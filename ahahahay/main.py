import pandas as pd
from MyLib.Utilities import IndoBertModelRep


lenData = 0


# mengambil informasi terkait jumlah data
while True:
    test = input("Masukan nomor data yang di load -> ingat semakin besar data semakin besar komputasi\t")
    try:
        res = int(test)
        lenData += res
        break
    except:
        print("Input Data Salah pastikan input anda berupa nominal tanpa koma!!")

baseData = pd.read_csv("./data/dataset.csv")
listOfData = baseData["Isi"].tolist()[:lenData]


model = IndoBertModelRep(listOfData)
print("Model Training.....")
model.fit_transform(lenData//3)
print("Selesai Training....")



while True:
    print("Silahkan pilih menu :\n")
    option = input("[1] : untuk melihat kemiripan pasal pilih [0] : untuk keluar\t")
    if(int(option) == 1):
        question = input("Silahkan masukan pertanyaan...\n")
        result = model.transformSentence(question)
        similar = result["result"][:3]
        besaran = [result["index"][a] for a in similar]
        pasal = [listOfData[a] for a in similar]
        for i,j in enumerate(pasal):
            print(f"pasal dengan bunyi {j} \n dengan kemiripan {besaran[i]}")

            
    elif(int(option) == 0):
        print("program dihentikan")
        break
    else:
        print("argumen tidak ditemukan")
    
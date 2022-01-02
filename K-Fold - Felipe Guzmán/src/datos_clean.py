#Inteligencia Artificial
#Felipe Guzman
import os
import pandas as pd
import numpy as np
os.chdir("../input")
df = pd.read_csv("steam.csv", dtype ={"positive_ratings":"int32","negative_ratings":"int32"})

df2 = pd.DataFrame(df,
                columns = ["appid","name","release_date","english","developer","publisher","plataforms","required_age","categories",
                "genres","steamspy_tags","achievements","gusta o no gusta","average_playtime",
                "median_playtime","owners","price"])


for x in df.index:
    if df.loc[x, "positive_ratings"] > df.loc[x, "negative_ratings"]:
        df2.loc[x, "gusta o no gusta"] = 1
    elif df.loc[x, "positive_ratings"] < df.loc[x, "negative_ratings"]:
        df2.loc[x, "gusta o no gusta"] = 0

df2 = df2.dropna(subset = ["gusta o no gusta"])

df3 = pd.DataFrame(df2,
                columns = ["appid",
                "release_date","english","developer","publisher"
                ,"required_age","categories",
                "genres","steamspy_tags","achievements","gusta o no gusta",
                "median_playtime","owners","price"])

df2["gusta o no gusta"] = pd.to_numeric(df2["gusta o no gusta"], downcast = "integer")
df3["gusta o no gusta"] = pd.to_numeric(df3["gusta o no gusta"], downcast = "integer")
df3["price"] = pd.to_numeric(df3["price"], downcast = "float")

df4 = pd.DataFrame(columns = ["appid","release_date","english","developer","publisher","required_age","categories",
                "genres","steamspy_tags","achievements","gusta o no gusta",
                "median_playtime","owners","price"])

#llenamos DF4 con los valores de las mil muestras 
contador_1 = 0
contador_2 = 0
for x in range(0,len(df3)-1):
    dato = df3.sample(n=1)
    for i in dato.index:
        if dato.loc[i, "gusta o no gusta"] == 0:
            if contador_1 < 500:
                df4 = df4.append(dato, ignore_index= True)
                contador_1 += 1
        elif dato.loc[i, "gusta o no gusta"] == 1:
            if contador_2 < 500:
                df4 = df4.append(dato, ignore_index= True)
                contador_2 += 1
        contador = contador_1 + contador_2
        if contador == 1000:
            break

owners = ["0-20000","20000-50000","50000-100000","100000-200000","200000-500000","500000-1000000","1000000-2000000","2000000-5000000","5000000-10000000","10000000-20000000"]
"""for i in range(len(owners)):
    print(owners[i])"""
for x in df4.index:
    for y in range(len(owners)):
        if df4.loc[x, "owners"] == owners[y]:
            df4.loc[x, "owners"] = y

media_price = np.mean(df4["price"])
std_price = np.std(df4["price"])
media_playtime = np.mean(df4["median_playtime"])
std_playtime = np.std(df4["median_playtime"])
media_achievements = np.mean(df4["achievements"])
std_achievements = np.std(df4["achievements"])
media_owners = np.mean(df4["owners"])
std_owners = np.std(df4["owners"])
media_english = np.mean(df4["english"])
std_english = np.std(df4["english"])
media_required_age = np.mean(df4["required_age"])
std_required_age = np.std(df4["required_age"])

for x in df4.index:
    df4.loc[x, "price"] = (float(df4.loc[x, "price"]) - media_price) / std_price
    df4.loc[x, "median_playtime"] = (float(df4.loc[x, "median_playtime"]) - media_playtime) / std_playtime
    df4.loc[x, "achievements"] = (float(df4.loc[x, "achievements"]) - media_achievements) / std_achievements
    df4.loc[x, "owners"] = (float(df4.loc[x, "owners"]) - media_owners) / std_owners
    df4.loc[x, "english"] = (float(df4.loc[x, "english"]) - media_english) / std_english
    df4.loc[x, "required_age"] = (float(df4.loc[x, "required_age"]) - media_required_age) / std_required_age

df5 = pd.DataFrame(df4, columns = ["english","required_age","achievements",
                "median_playtime","owners","price","gusta o no gusta"])

#KFlod 
#particion del dataset en 5 folds
folds = np.array_split(df5.sample(frac=1), 5)

#iteraciones de los folds con los indices 0 de tt_1...5 que son los datos de prueba y los indices 1...4 de tt_1...5 son de entrenamiento
tt_1 = [folds[0], folds[1], folds[2], folds[3], folds[4]]
tt_2 = [folds[1], folds[0], folds[2], folds[3], folds[4]]
tt_3 = [folds[2], folds[0], folds[1], folds[3], folds[4]]
tt_4 = [folds[3], folds[0], folds[1], folds[2], folds[4]]
tt_5 = [folds[4], folds[0], folds[1], folds[2], folds[3]]

conjuntos=[tt_1, tt_2, tt_3, tt_4, tt_5]

ite1 = pd.DataFrame(columns = ["KFOLD"])
ite2 = pd.DataFrame(columns = ["KFOLD"])
ite3 = pd.DataFrame(columns = ["KFOLD"])
ite4 = pd.DataFrame(columns = ["KFOLD"])
ite5 = pd.DataFrame(columns = ["KFOLD"])

entrenamiento1 = pd.concat([folds[1],folds[2],folds[3],folds[4]],axis=0)

ite1 = ite1.append(folds[0])
ite1["KFOLD"].fillna("PRUEBA", inplace=True)
ite1 = ite1.append(entrenamiento1)
ite1["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)

entrenamiento2 = pd.concat([folds[0],folds[2],folds[3],folds[4]],axis=0)

ite2 = ite2.append(folds[1])
ite2["KFOLD"].fillna("PRUEBA", inplace=True)
ite2 = ite2.append(entrenamiento2)
ite2["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)
    
entrenamiento3 = pd.concat([folds[0],folds[1],folds[3],folds[4]],axis=0)

ite3 = ite3.append(folds[2])
ite3["KFOLD"].fillna("PRUEBA", inplace=True)
ite3 = ite3.append(entrenamiento2)
ite3["KFOLD"].fillna("ENTRENAMIENTO", inplace=True) 

entrenamiento4 = pd.concat([folds[0],folds[1],folds[2],folds[4]],axis=0)

ite4 = ite4.append(folds[3])
ite4["KFOLD"].fillna("PRUEBA", inplace=True)
ite4 = ite4.append(entrenamiento2)
ite4["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)

entrenamiento5 = pd.concat([folds[0],folds[1],folds[2],folds[3]],axis=0)

ite5 = ite5.append(folds[4])
ite5["KFOLD"].fillna("PRUEBA", inplace=True)
ite5 = ite5.append(entrenamiento2)
ite5["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)

   
os.chdir("../output")
#df2.to_csv("steam_clean.csv", index = None, header = True, encoding = "utf-8-sig", sep = ";")
df4.to_csv("steam_1000_muestras_aleatorias.csv", index = None, header = True, encoding = "utf-8-sig", sep = ";")
df5.to_csv("steam_1000_muestras_normalizado.csv",index = None, header = True, encoding = "utf-8-sig", sep = ";")
with pd.ExcelWriter('KFOLD-1000-muestras.xlsx') as writer:
    ite1.to_excel(writer, sheet_name="Iteracion 1")
    ite2.to_excel(writer, sheet_name="Iteracion 2")
    ite3.to_excel(writer, sheet_name="Iteracion 3")
    ite4.to_excel(writer, sheet_name="Iteracion 4")
    ite5.to_excel(writer, sheet_name="Iteracion 5")
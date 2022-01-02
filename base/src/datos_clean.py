#Inteligencia Artificial
#Felipe Guzman
import os
import pandas as pd
import numpy as np
os.chdir("../input")
df = pd.read_csv("steam.csv", dtype ={"positive_ratings":"int32","negative_ratings":"int32"})

def Kmeans(dfcompleto):
    df_origin = pd.DataFrame()
    df_origin = dfcompleto.reset_index(drop = True) #se resetea los indices 
    #para que guardar las clases y no ser eliminadas con el drop (problema que surgió)

    dfcompleto.drop(['gusta o no gusta'], axis='columns', inplace=True) #se elimina la clase
    
    df_new= dfcompleto
    centroides_prueba=df_new.sample(n=2) #obtengo mis 2 centroides de prueba de manera al azar
    centroides_prueba= centroides_prueba.reset_index(drop=True) 
    
    comp_centroide1 = pd.DataFrame() #creamos un dataframe 
    #para almacenar las comparaciones del centroide 1
    comp_centroide2 = pd.DataFrame() #creamos un dataframe 
    #para almacenar las comparaciones del centroide 2
    
    cloud1= pd.DataFrame() #creamos un dataframe para la cloud1
    cloud2= pd.DataFrame() #creamos un dataframe para la cloud2

    m = 0

        
    while(m<20): #numero de veces que se recalculan los centroides
        for i in range(len(df_new)):
            dist1=(np.linalg.norm(centroides_prueba.loc[0]-df_new.loc[i])) #se calcula la distancia euclidiana entre los centroides y
            dist2=(np.linalg.norm(centroides_prueba.loc[1]-df_new.loc[i])) #lo cada uno de los parametros

            if dist1<dist2:
                cloud1=cloud1.append(df_new.loc[i]) #se almacena los parametros en la nube que corresponda
            elif dist2<dist1:
                cloud2=cloud2.append(df_new.loc[i]) #se almacena los parametros en la nube que corresponda
                
        print("\nCentroides terminados\n")
        
        centroide_i1 = cloud1.mean() #se calcula un nuevo centroide con la media de la nube 1
        centroide_i2 = cloud2.mean() #se calcula un nuevo centroide con la media de la nube 2
        
        nube1 = pd.DataFrame()
        nube2 = pd.DataFrame()
        
        for i in range(len(df_new)):
            dist1 = (np.linalg.norm(centroide_i1-df_new.loc[i])) #se vuelve a calcular la distancia euclidiana con el nuevo centroide a cada uno de los parametros
            dist2 = (np.linalg.norm(centroide_i2-df_new.loc[i])) #se vuelve a calcular la distancia euclidiana con el nuevo centroide a cada uno de los parametros
            
            if dist1<dist2:
                nube1=nube1.append(df_new.loc[i]) #se crea la nueva nube con los parametros calculados del nuevo centroide
            elif dist2<dist1:
                nube2=nube2.append(df_new.loc[i]) #se crea la nueva nube con los parametros calculados del nuevo centroide
        
        comp_centroide1 = comp_centroide1.append(centroides_prueba.loc[0])
        comp_centroide1 = comp_centroide1.append(centroide_i1, ignore_index=True) #se comparan los centroides 1 y 2
        comp_centroide2 = comp_centroide2.append(centroides_prueba.loc[1])
        comp_centroide2 = comp_centroide2.append(centroide_i2, ignore_index=True)
        
        desviacion1 = comp_centroide1.std() #calculamos la desviacion estandar de los centroides
        desviacion2 = comp_centroide2.std()
        
        desviacion1 = desviacion1.mean(axis=0) #se muestra una media de la desviacion estandar de cada centroide
        desviacion2 = desviacion2.mean(axis=0)
        
        print(desviacion1)
        print(desviacion2)
        
        
        m += 1
    print("\n")
    print("datos en la nube 1: ",len(nube1))
    print("datos en la nube 2: ",len(nube2))
    
    comprobacion(nube1, nube2, df_origin) #se llama a la funcion de comprobacion de las nubes con sus patrones
    

def comprobacion(nube1, nube2, df_origin):
    acierto = 0
    errado = 0
    
    for i in range(len(nube1)): 
        for j in range(len(df_origin)):
            if nube1.index[i] == df_origin.index[j]: #se comparan los indices dela nube 1 con los indices de la tabla original, para encontrar la clase de los parametros
                indice = df_origin.loc[j]              #de la nube 1
                if indice["gusta o no gusta"] == 1:
                    acierto += 1
                elif indice["gusta o no gusta"] == 0:
                    errado += 1
    
    peracierto = (acierto*100)/len(nube1)
    pererrado = (errado*100)/len(nube1)
    
    print("\n")
    print("Aciertos Nube 1: ", acierto, "Porcentaje: %",round(peracierto))
    print("Errados Nube 1: ",errado, "Porcentaje: %",round(pererrado))
    print("\n")
    
    acierto = 0
    errado = 0
       
    for i in range(len(nube2)):
        for j in range(len(df_origin)):
            if nube2.index[i] == df_origin.index[j]: #se comparan los indices dela nube 2 con los indices de la tabla original, para encontrar la clase de los parametros
                indice = df_origin.loc[j] #de la nube 2
                if indice["gusta o no gusta"] == 1:
                    acierto += 1
                elif indice["gusta o no gusta"] == 0:
                    errado += 1
                    
    peracierto = (acierto*100)/len(nube2)
    pererrado = (errado*100)/len(nube2)
    
    print("Aciertos Nube 2: ", acierto, "Porcentaje: %",round(peracierto))
    print("Errados Nube 2: ",errado, "Porcentaje: %",round(pererrado))


def KNN(k,t,e,td):
    prueba = t.reset_index(drop=True)
    entrenamiento = e.reset_index(drop=True)
    total_datos = td.reset_index(drop=True) #se resetean los indices de los datos
    test=t
    test = test.reset_index()
    test.drop(["gusta o no gusta"], axis = "columns", inplace=True) #eliminamos la clase de las pruebas
    train=e
    train = train.reset_index()
    train.drop(["gusta o no gusta"], axis = "columns", inplace=True) #eliminamos la clase de los entrenamientos
    df = pd.DataFrame(columns = ["Distancia"])
    short_dis = pd.DataFrame(columns = ["Distancia", "Clase"])
    
    aciertos = []
    indice = 0
    cont_aciertos = 0
    cont_fallidas = 0  

    for i in range(len(test)): #recorremos los 200 datos de las pruebas
        for j in range(len(train)): #recorremos los 800 datos de los entrenamientos
            dis = np.linalg.norm(test.iloc[i]-train.iloc[j]) #calculamos la distancia euclidiana 
                                                             #con cada parametro de las pruebas y los entrenamientos
            df = df.append({"Distancia": dis}, ignore_index=True) #agregamos al dataframe que almacena la distancia
            indice += 1

        df["Distancia"] = pd.to_numeric(df["Distancia"], downcast="float")
        df = df.sort_values(by=["Distancia"]) #ordenamos de menor a mayor las distancias
        short_dis = short_dis.append(df.head(k)) #obtenemos los K primeras distancias que son las mas cortas

        indice_short_dis = short_dis.index.tolist()
        indice_total_datos = total_datos.index.tolist()

        contador_0 = 0
        contador_1 = 0

        for l in range(len(short_dis)):
            for m in range(len(indice_total_datos)):
                if indice_short_dis[l] == indice_total_datos[m]: #comparamos los indices para verificar la clase 
                    if total_datos.loc[m, "gusta o no gusta"] == 1:
                        contador_1 += 1
                    elif total_datos.loc[m, "gusta o no gusta"] == 0:
                        contador_0 += 1
                     
        if contador_0 > contador_1: 
            if total_datos.loc[i, "gusta o no gusta"] == 0:
                cont_aciertos +=1
            elif total_datos.loc[i, "gusta o no gusta"] == 1:
                cont_fallidas +=1
                
        elif contador_1 > contador_0:
            if total_datos.loc[i, "gusta o no gusta"] == 1:
                cont_aciertos +=1
            elif total_datos.loc[i, "gusta o no gusta"] == 0:
                cont_fallidas +=1

        
        df_idx = df[df["Distancia"]<=1000].index #estas lineas son para borrar los datos del dataframe con las distancias antiguas y colocar las nuevas
        df = df.drop(df_idx)
        short_dis_idx = short_dis[short_dis["Distancia"]<=1000].index
        short_dis = short_dis.drop(short_dis_idx)

    print("Porcentaje Acierto Local: ",(cont_aciertos*100)/len(test)) #calculamos el porcentaje de acierto local
    aciertos = (cont_aciertos*100)/len(test)


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

#Las variables se cambian a valores netamente numericos, para luego sacar su promedio y desviación estandar.
#Con estos valores podemos normalizar el dataset.

owners = ["0-20000","20000-50000","50000-100000","100000-200000","200000-500000",
        "500000-1000000","1000000-2000000","2000000-5000000",
        "5000000-10000000","10000000-20000000"]
for x in df4.index:
    for y in range(len(owners)):
        if df4.loc[x, "owners"] == owners[y]: #Reemplazamos los valores de tipo String 
            df4.loc[x, "owners"] = y #y los volvemos valores enteros para luego normalizar.

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
nuevo_df = df5.sample(frac=1)

#iteraciones de los folds con los indices 0 de tt_1...5 que son los datos de prueba y 
# los indices 1...4 de tt_1...5 son de entrenamiento
tt_1 = [folds[0], folds[1], folds[2], folds[3], folds[4]]
tt_2 = [folds[1], folds[0], folds[2], folds[3], folds[4]]
tt_3 = [folds[2], folds[0], folds[1], folds[3], folds[4]]
tt_4 = [folds[3], folds[0], folds[1], folds[2], folds[4]]
tt_5 = [folds[4], folds[0], folds[1], folds[2], folds[3]]

conjuntos=[tt_1, tt_2, tt_3, tt_4, tt_5]

tests = [folds[0], folds[1], folds[2], folds[3], folds[4]]

total_datos = pd.concat([folds[0],folds[1],folds[2],folds[3],folds[4]],axis=0) 

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

Training = [entrenamiento1,entrenamiento2,entrenamiento3,entrenamiento4,entrenamiento5]


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
    
k = int(input("Ingrese valor K: ")) #entregamos el valor de K ingresado por el usuario, tiene que ser 3,5 o 7

for x in range(len(tests)): #utilizamos la función knn con los 5 folds creados
    KNN(k,tests[x],Training[x],total_datos)
    
Kmeans(nuevo_df) #si va a utilizar la funcion K means, le recomiendo comentar la sección de codigo de las lineas 337 hasta la 340, ya que pertenecen a K-Fold
    
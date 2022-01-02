#Inteligencia Artificial
#Felipe Guzmán
import numpy as np
import pandas as pd
import os
import tensorflow as tf


os.chdir("../output")
df = pd.read_csv("steam_1000_muestras_normalizado.csv",sep = ";")

contador = 0

def sigmoide(x):
    return 1/(1 + np.exp(-x))

def red_neuronal(entrenamiento):
    global contador
    contador += 1
    input = np.array([],dtype=float)#se crea un array vacio 
    out = np.array([],dtype=float)#se crea un array vacio 
    original = pd.DataFrame()#Creación de dataframes
    entrada = pd.DataFrame()
    salida = pd.DataFrame()
    comparacion = pd.DataFrame(columns = ["Clase Original", "Clase Predicha"])#Dataframe para comprar las clases
    original = entrenamiento.reset_index(drop = True)#se resetean los indices
    entrada = entrenamiento.reset_index(drop = True)#para evitar conflictos
    salida = entrenamiento.reset_index(drop = True)
    entrada.drop(["gusta o no gusta"], axis = "columns", inplace = True)#Se elimina la clase para realizar
                                                                        #las predicciones
    
    #Se transforman los dataframes en arreglos, con sus respectivos datos
    for x in entrada.index:
        input = np.append(input,entrada.loc[x].std()) #se calcula la desviación estandar
        #para ser calculado por las neuronas

    salida  = entrenamiento["gusta o no gusta"]
    for i in salida.index:
        out = np.append(out, salida.loc[i]) #se almacena el dataframe en el arreglo

    #Se crean 2 neuronas en la capa oculta, y se le entregan los valores de entrada
    capa_oculta_1 = tf.keras.layers.Dense(units = 2, input_shape=[1]) #es 1 solo arreglo de 800 datos
    output = tf.keras.layers.Dense(units = 1) #se crea 1 neurona para el procesamiento de salida
    modelo = tf.keras.Sequential([capa_oculta_1, output]) #se utiliza el modelo secuencial para trabajar 
    #con las capas.
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.05), #utilizamos el optimizador Adam, y una taza de aprendizaje de 0.05
        loss='mean_squared_error'
    )
    
    print("Comenzando entrenamiento",contador)
    historial = modelo.fit(input, out, epochs=500, verbose=False) #Entrenamiento de los datos, con 500 vueltas
    print("Modelo entrenado!")
    
    print("Realizando predicciones...")
    for j in original.index: #ciclo donde se recorre los datos originales
        resultado = modelo.predict([input[j]]) #realiza la prediccion con los datos ya entrenados
        comparacion.loc[j, "Clase Predicha"] = resultado #guarda los valores en el dataframe comparacion
        comparacion.loc[j, "Clase Original"] = original.loc[j, "gusta o no gusta"]#guarda los valores en el dataframe comparacion
    
    print(comparacion)
    
    
    
    if contador == 1:
        archivo_salida1(comparacion)
    elif contador == 2:
        archivo_salida2(comparacion)
    elif contador == 3:
        archivo_salida3(comparacion)
    elif contador == 4:
        archivo_salida4(comparacion)
    elif contador == 5:
        archivo_salida5(comparacion)
        
def archivo_salida1(comparacion):
    os.chdir("../output/Comparaciones")
    comparacion.to_csv("Comparacion Folds 1.csv",index = None, header = True, encoding = "utf-8-sig", sep = ";")
def archivo_salida2(comparacion):
    comparacion.to_csv("Comparacion Folds 2.csv",index = None, header = True, encoding = "utf-8-sig", sep = ";")
def archivo_salida3(comparacion):
    comparacion.to_csv("Comparacion Folds 3.csv",index = None, header = True, encoding = "utf-8-sig", sep = ";")
def archivo_salida4(comparacion):
    comparacion.to_csv("Comparacion Folds 4.csv",index = None, header = True, encoding = "utf-8-sig", sep = ";")
def archivo_salida5(comparacion):
    comparacion.to_csv("Comparacion Folds 5.csv",index = None, header = True, encoding = "utf-8-sig", sep = ";")


folds = np.array_split(df.sample(frac=1), 5)

tt_1 = [folds[0], folds[1], folds[2], folds[3], folds[4]]
tt_2 = [folds[1], folds[0], folds[2], folds[3], folds[4]]
tt_3 = [folds[2], folds[0], folds[1], folds[3], folds[4]]
tt_4 = [folds[3], folds[0], folds[1], folds[2], folds[4]]
tt_5 = [folds[4], folds[0], folds[1], folds[2], folds[3]]

conjuntos=[tt_1, tt_2, tt_3, tt_4, tt_5]

total_datos = pd.concat([folds[0],folds[1],folds[2],folds[3],folds[4]],axis=0) #######

entrenamiento1 = pd.concat([folds[1],folds[2],folds[3],folds[4]],axis=0)

tests = [folds[0], folds[1], folds[2], folds[3], folds[4]] ##################

df1 = pd.DataFrame(columns = ["KFOLD"])
df2 = pd.DataFrame(columns = ["KFOLD"])
df3 = pd.DataFrame(columns = ["KFOLD"])
df4 = pd.DataFrame(columns = ["KFOLD"])
df5 = pd.DataFrame(columns = ["KFOLD"])


 
df1 = df1.append(folds[0])
df1["KFOLD"].fillna("PRUEBA", inplace=True)
df1 = df1.append(entrenamiento1)
df1["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)

entrenamiento2 = pd.concat([folds[0],folds[2],folds[3],folds[4]],axis=0)
#print("Prueba 2: ",tt_2[1].index,"\nEntrenamiento 3: ",entrenamiento1.index)

df2 = df2.append(folds[1])
df2["KFOLD"].fillna("PRUEBA", inplace=True)
df2 = df2.append(entrenamiento2)
df2["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)
    
entrenamiento3 = pd.concat([folds[0],folds[1],folds[3],folds[4]],axis=0)
#print("Prueba 3: ",tt_3[2].index,"\nEntrenamiento 3: ",entrenamiento1.index)

df3 = df3.append(folds[2])
df3["KFOLD"].fillna("PRUEBA", inplace=True)
df3 = df3.append(entrenamiento2)
df3["KFOLD"].fillna("ENTRENAMIENTO", inplace=True) 

entrenamiento4 = pd.concat([folds[0],folds[1],folds[2],folds[4]],axis=0)
#print("Prueba 4: ",tt_4[3].index,"\nEntrenamiento 4: ",entrenamiento1.index) 

df4 = df4.append(folds[3])
df4["KFOLD"].fillna("PRUEBA", inplace=True)
df4 = df4.append(entrenamiento2)
df4["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)

entrenamiento5 = pd.concat([folds[0],folds[1],folds[2],folds[3]],axis=0)
#print("Prueba 5: ",tt_5[4].index,"\nEntrenamiento 5: ",entrenamiento1.index) 

df5 = df5.append(folds[4])
df5["KFOLD"].fillna("PRUEBA", inplace=True)
df5 = df5.append(entrenamiento2)
df5["KFOLD"].fillna("ENTRENAMIENTO", inplace=True)

Training = [entrenamiento1,entrenamiento2,entrenamiento3,entrenamiento4,entrenamiento5]#########
  

nuevo_df = df.sample(frac=1)

for x in range(len(Training)):
    red_neuronal(Training[x])



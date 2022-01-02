#Inteligencia Artificial
#Felipe Guzmán
import numpy as np
import pandas as pd
import os


#os.chdir("../output")
df = pd.read_csv("steam_1000_muestras_kflod.csv",sep = ";")

def sigmoide(x):
    return 1/(1 + np.exp(-x))
def red_neuronal(entrenamiento):
    valores = entrenamiento.drop("gusta o no gusta", axis = 1) #Se elimina la clase
    objetivo = entrenamiento["gusta o no gusta"] #Obejtivo solo la clase 
    #Hyperparametros
    capa_escondida = 2 #Numero de unidades en la capa escondida
    epochs = 1000 #Numero de iteraciones sobre el conjunto de entrenamiento
    taza_aprendizaje = 0.05 #Taza de aprendizaje

    ult_costo = None

    k,m = entrenamiento.shape # Numero de parametros de entrenamiento, numero de dimensiones de los datos
    print(k)
    #Inicializacion de los pesos
    entrada_escondida = (np.random.rand(k,capa_escondida)*0.98)+0.01
    print(entrada_escondida)
    salida_escondida = (np.random.rand(capa_escondida)*0.98)+0.01

    #entrenamiento
    for e in range(epochs):
        #variables para el gradiente
        gradiente_entrada_escondida = np.zeros(entrada_escondida.shape)
        gradiente__salida_escondida = np.zeros(salida_escondida.shape)
        
    #itera sobre el conjunto de entrenamiento
        
    #X es la entrada, Y es la salida
    for x,y in zip(valores.values, objetivo):
        z = sigmoide(np.matmul(x, entrada_escondida))
        w = sigmoide(np.matmul(salida_escondida, z)) #prediccion
        
        #pasada hacia atras (backward pass)
        salida_error = (y - w) * w * (1 - w)
        escondida_error = np.dot(salida_error, escondida_error) * z * (1-z)
        
        gradiente_entrada_escondida += escondida_error * x[:,None]
        gradiente__salida_escondida += salida_error * z
        
        print("Prediccion: ",w)
        print("Error: ",escondida_error)
        
    #es lo que pude realizar, pero tengo muchos errores con el manejo de datos de los dataframe
            
"""    #Actualiza los parámetros (pesos)
    entrada_escondida += taza_aprendizaje * gradiente_entrada_escondida / m
    salida_escondida += taza_aprendizaje * gradiente__salida_escondida / m
    
    if e % (epochs / 10) == 0:
        z = sigmoide(np.dot(valores.values, entrada_escondida))
        w = sigmoide(np.dot(z, salida_escondida))
        
        #Función de costo
        costo = np.mean((w - objetivo)**2)
        
        if ult_costo and ult_costo < costo:
             print("Costo de entrenamiento: ",costo)
        else:
            print("Costo de entrenamiento: ",costo)
            
        ult_costo = costo"""

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

red_neuronal(entrenamiento1)



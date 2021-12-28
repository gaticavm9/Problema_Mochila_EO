import sys
import time
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import random 

#sys.argv = ['mochila.py','1','1.4','1000','dat_3_200_1.txt']

if len(sys.argv) == 5:
    semilla = int(sys.argv[1])
    tau = float(sys.argv[2])
    ite = int(sys.argv[3])
    entrada = sys.argv[4]
    print('Parametros de entrada:',semilla, tau, ite, entrada,'\n')
else:
    print("Error en la entrada de los parámetros")
    print("Los parametros a ingresar son: Semilla, Tau, NroIteraciones, DatosEntrada")
    sys.exit(0)

tiempo_proceso_ini = time.process_time()
np.random.seed(semilla)

datosArchivo = pd.read_table(entrada, header=None, skiprows=2, nrows=2, sep=" ", names=range(2))
capacidad = int(datosArchivo.drop(columns=0,axis=1).to_numpy()[0])
valorOptimo = int(datosArchivo.drop(columns=0,axis=1).to_numpy()[1])
##Llenar una matriz con los valores del archivo   (ex matrizCoordenadas)
matrizElementos = pd.read_table(entrada, header=None, skiprows=5, sep=",", names=range(4))
matrizElementos = matrizElementos.drop(index=(len(matrizElementos)-1),axis=0)
matrizElementos = matrizElementos.drop(columns=0,axis=1).to_numpy()
numVariables = matrizElementos.shape[0]
#print('Matriz de Elementos:\n', matrizElementos,'\ntamaño',matrizElementos.shape, '\ntipo',type(matrizElementos))
#print('Número de variables:', numVariables,'\n')

#######################################
############   Funciones   ############
#######################################
#Función para calcular Ganancia mochila
#n: num de items      #mE: matriz detalle de Elementos      #moc: mochila a evaluar  
def funCalculaGanancia(n,mE,moc):
    aux = 0
    for i in range(n):
        aux += (mE[i][0]) * moc[i]
    return aux
#Función para calcular peso mochila
def funCalculaPeso(n,mE,moc):
    aux = 0
    for i in range(n):
        aux += (mE[i][1]) * moc[i]
    return aux
#Función para Evaluar Solucion
def funEvalSol(n,mE,sol):
    if(funCalculaPeso(n,mE,sol) <= capacidad):   # 53484
        fact = True
    else:
        fact = False  
    return fact

#Función generadora vector Probabilidad
def funVectorProb(n,tau):
    vPb = np.zeros(n)
    for i in range(n):
        vPb[i] = (i+1)**(-tau)
    return vPb
"""    
#Función generadora vector Proporciones
def funVectorProp(n,vPb):
    sumProb=np.sum(vPb)
    vPp = np.zeros(n)
    for i in range(n):
            vPp[i] = vPb[i]/sumProb
    return vPp     
#Función generadora vector Ruleta
def funVectorRuleta(n,vPp):
    vR = np.zeros(n)
    for i in range(n):
            vR[i] = vR[i-1] + vPp[i]
    return vR        
"""
#Función Fitness
def funFitness(n,mE):
    fit = np.zeros((n,2))
    for i in range(n):
            fit[i][0] = mE[i][0] / mE[i][1]
            fit[i][1] = i
    return fit 
#Función Ordenar Fitness Sol Factible
def funFitnessFactible(sol,fit):
    n=sol.size
    fitOrd = np.zeros((n,2))
    for i in range(n):  
        if sol[i] == 0:
            fitOrd[i][0]= fit[i][0]  #Agrega los fitnes de los items que no estan en mochila
            fitOrd[i][1]= fit[i][1]   #identificador del item         
    #Elimina 0
    fitOrd = np.delete(fitOrd, np.where(fitOrd[:, 0] == 0)[0], axis=0)
    #ordena
    fitOrd = fitOrd[fitOrd[:, 0].argsort()]
    fitOrd=fitOrd[::-1]  #Mayor a Menor       
    return fitOrd   

#Función Ordenar Fitness Sol NO Factible
def funFitnessNoFac(sol,fit):
    n=sol.size
    fitOrd = np.zeros((n,2))
    for i in range(n):  
        if sol[i] == 1:   #1
            fitOrd[i][0]= fit[i][0]  #Agrega los fitnes de los items que no estan en mochila
            fitOrd[i][1]= fit[i][1]   #identificador del item         
    #Elimina 0
    fitOrd = np.delete(fitOrd, np.where(fitOrd[:, 0] == 0)[0], axis=0)
    #ordena
    fitOrd = fitOrd[fitOrd[:, 0].argsort()]
    return fitOrd

#Función Crear Ruleta
def funCrearRuleta(n):
    vR = np.zeros(n)
    vPp = np.zeros(n)
    sumProb = 0.0
    for i in range(n):
        sumProb = sumProb + vProb[i]
    for i in range(n):
            vPp[i] = vProb[i]/sumProb
    #vR[0] = Vp[0]        
    for i in range(n):
            vR[i] = vR[i-1] + vPp[i]     
    return vR
#Función Girar Ruleta
def funGirarRuleta(vR):
    aux=0
    listo = False
    giro = 0.0
    giro = random.uniform(0, 1)    
    while listo==False:
        if giro <= vR[aux]:
            listo = True
        else:
            aux = aux+1        
    return aux             
####################################################################


########################################
###   Implementación del algoritmo   ###
########################################
generacion=0
solFactible=False
###1: Generar una solucion inicial randomica
solucion =  np.random.randint(2, size=numVariables) #Llena vector con 0 y 1
#Evaluar
solFactible = funEvalSol(numVariables,matrizElementos,solucion)
###2: Se asigna como mejor solución
if(solFactible == True): #Si la solucion inicial no es valida, se llena con -1
    solucionMejor = solucion
else:
    solucionMejor = solucion
    solucionMejor = np.zeros(numVariables)
gananciaSolMejor = funCalculaGanancia(numVariables,matrizElementos,solucionMejor)
"""
print(solucionMejor, solucionMejor.size)
print("Ganancia solucionMejor ", funCalculaGanancia(numVariables,matrizElementos,solucionMejor))
print("Peso solucionMejor ", funCalculaPeso(numVariables,matrizElementos,solucionMejor))
###
print(solucion, solucionMejor.size)
print("Ganancia solucionMejor ", funCalculaGanancia(numVariables,matrizElementos,solucion))
print("Peso solucionMejor ", funCalculaPeso(numVariables,matrizElementos,solucion))
"""
###3: Generar el vector de probabilidades P segun la ecuación               44:20
vProb = funVectorProb(numVariables, tau)
# Vector Proporción #vProp = funVectorProp(numVariables, vProb) #print(vProp, vProp.size, "suma proporciones ",np.sum(vProp))
# Vector Ruleta#vRuleta = funVectorRuleta(numVariables, vProp) #print(vRuleta, vRuleta.size, "suma proporciones ",np.sum(vRuleta))

#Vector con valores Fitness para cada item
vFit = funFitness(numVariables,matrizElementos) #print("Valores Fitness: \n",vFit, vFit.size)

###4: For a ite number of iterations do                 44:45
while generacion<ite and gananciaSolMejor<valorOptimo:    ## generacion < ite:
    generacion+=1
    #Generar fitness para solucion
    if solFactible==True:  #True
        vFitOrdenado = funFitnessFactible(solucion, vFit)
    else:
        vFitOrdenado = funFitnessNoFac(solucion, vFit)
    #Generar Ruleta de acuerdo al tamano del Fitness Ordenado
    vRuleta = funCrearRuleta(int(vFitOrdenado.size/2))
    #Seleccionar item de la ruleta
    itemSelec = funGirarRuleta(vRuleta)

    #Cambiar el valor en solucion del item seleccionado
    if int(solucion[int(vFitOrdenado[itemSelec][1])])==0:
        solucion[int(vFitOrdenado[itemSelec][1])]=1
    else:
        solucion[int(vFitOrdenado[itemSelec][1])]=0
    
    #Evaluar nueva solucion
    solFactible = funEvalSol(numVariables,matrizElementos,solucion)
    gananciaSol = funCalculaGanancia(numVariables,matrizElementos,solucion)
    #Si la nueva solucion es Factible y mejor que la solucionMejor actual, se actualiza
    if (solFactible==True and gananciaSol>gananciaSolMejor):
        solucionMejor = solucion
        gananciaSolMejor = funCalculaGanancia(numVariables,matrizElementos,solucionMejor)
# ------------ termino ciclo While -------------    
### Salida
print("Ganancia de mejor solucion ",gananciaSolMejor)
print("Iteraciones realizadas ",generacion)
print("Mejor solucion: \n",solucion)

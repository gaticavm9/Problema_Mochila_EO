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
    if(funCalculaPeso(n,mE,sol) <= capacidad):
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
print("Factible? ", solFactible)
###2: Se asigna como mejor solución
if(solFactible == True): #Si la solucion inicial no es valida, se llena con -1
    solucionMejor = solucion
else:
    solucionMejor = solucion
    solucionMejor = np.zeros(numVariables)
"""
print(solucionMejor, solucionMejor.size)
print("Ganancia solucionMejor ", funCalculaGanancia(numVariables,matrizElementos,solucionMejor))
print("Peso solucionMejor ", funCalculaPeso(numVariables,matrizElementos,solucionMejor))
###
print(solucion, solucionMejor.size)
print("Ganancia solucionMejor ", funCalculaGanancia(numVariables,matrizElementos,solucion))
print("Peso solucionMejor ", funCalculaPeso(numVariables,matrizElementos,solucion))
"""
###3: Generar el vector de probabilidades P segun la ecuación  44:20
vProb = funVectorProb(numVariables, tau)
# Vector Proporción 
vProp = funVectorProp(numVariables, vProb) #print(vProp, vProp.size, "suma proporciones ",np.sum(vProp))
# Vector Ruleta
vRuleta = funVectorRuleta(numVariables, vProp) #print(vRuleta, vRuleta.size, "suma proporciones ",np.sum(vRuleta))



###4: 

    


"""

#Se procede a crear colonia vacia (tamaño colonia x num variable) inicializado con el valor -1
colonia=np.full((col, numVariables), fill_value=-1, dtype=int)
#print('Colonia:\n',colonia, '\ntamaño:', colonia.shape, '\ntipo:', type(colonia),'\n')



#Creación primera solución y se asigna como mejor solución encontrada
solucionOptima = np.array([0,48,31,44,18,40,7,8,9,42,32,50,10,51,13,12,46,25,26,27,11,24,3,5,14,4,23,47,37,36,39,38,35,34,33,43,45,15,28,49,19,22,29,1,6,41,20,16,2,17,30,21,-2])
solucionMejor = np.arange(0,numVariables)
np.random.shuffle(solucionMejor)
solucionMejorCosto = solucionCalculaCosto(numVariables, solucionMejor, matrizDistancias)
solucionMejorIteracion=0
#print('Solucion inicial y a la vez mejor solucion:\n', solucionMejor,'\ntamaño:', solucionMejor.shape,'\ntipo',type(solucionMejor))
#print('Costo de la solucion inicial y a la vez mejor solucion: ',solucionMejorCosto)
#print('Iteración donde se encontró la mejor solución:', solucionMejorIteracion,'\n')

#Creación Matriz de feromona
matrizFeromona = np.full_like(matrizDistancias,fill_value=1/solucionMejorCosto,dtype=float)
T0= matrizFeromona[0][0]
#print('Matriz de Feromona: \n',matrizFeromona,'\ntamaño:',matrizFeromona.shape,'\ntipo:',type(matrizFeromona),'\n')


## Aplicación del algoritmo ACS
#Inicio ciclo iterativo de ACS por numero predefinido de iteraciones
generacion=0

##Funcion 1.1 selecciona el nodo siguiente a visitar para el caso 0 < x < q0
def proxNodo1(nodo, hormiga):
    FxH=0
    max=0
    pos=0
    for i in range(numVariables):
        if not i in colonia[hormiga]:  #Restringir que ya está visitado (Solo nodo actual if(i!=nodo):)
            FxH = matrizFeromona[nodo][i] * (matrizHeuristica[nodo][i] ** B)
            if(FxH > max):
                max = FxH
                pos = i  
    return pos
##Funcion 2.2 selecciona el nodo siguiente a visitar para el caso q0 < x < 1
def proxNodo2(nodo, hormiga):
    prob=[]
    posProb=[]
    pos2=0
    sumFxH=0
    #Sumatoria de FxH**B
    for i in range(numVariables):
        if not i in colonia[hormiga]:
            sumFxH= sumFxH + (matrizFeromona[nodo][i] * (matrizHeuristica[nodo][i] ** B))
    #Hallar probabilidades y guardarlas en una lista
    for i in range(numVariables):
        if not i in colonia[hormiga]:  #Restringir que ya está visitado (Solo nodo actual if(i!=nodo):)
            prob.append( (matrizFeromona[nodo][i] * (matrizHeuristica[nodo][i] ** B)) / sumFxH )
            posProb.append(i)
    #Seleccionar un elemento con ruleta
    selec=random.choices(prob, weights=(prob), k=1)[0]
    posAux = prob.index(selec)
    pos2 = posProb[posAux]     
    return pos2
#Funcion Actualizar Feromona local
def feromL(ii, jj):
    ferL = ((1-tev)*(matrizFeromona[ii][jj])) + tev*T0
    return ferL
#Funcion Actualizar Feromona Global
def feromGlob(nVar, sMej, mFer, sCos):
    for i in range(nVar):
        for j in range(nVar):
            #Busco indice donde está i en la mejor solucion, para comprobar si el elemento que sigue es igual a j (seria un segmento recorrido por la hormiga)
            indexI = np.where(sMej == i)
            indexI = int(indexI[0])
            if(indexI < numVariables-1): 
                if((sMej[indexI+1]) == j):
                    mFer[i][j] = ((1-tev)*mFer[i][j]) + (tev*(1/sCos))
                else:
                    mFer[i][j] = ((1-tev)*mFer[i][j]) + 0    
    return mFer    

########################################
###   Implementación del algoritmo   ###
########################################
while generacion < ite: ## generacion < ite:
    colonia=np.full((col, numVariables), fill_value=-1, dtype=int)
    generacion+=1
    print('G:',generacion)
    colonia[:, 0] =  np.random.randint(0, numVariables, size=(1, col)) #Llenar primera columna con posicion inicial de las hormigas np.random.randint(0, numVariables, size=(1, col))
    FxH=[]
    #Camino de Hormigas
    for i in range(numVariables):  #numVariables
        for j in range(col):  #col
            #Hormiga avanza                   
            if(np.random.random() <= q0):
                #Formula (1)
                if(i < numVariables-1):
                    colonia[j][i+1] =  proxNodo1(colonia[j][i], j)  #Colocar proximo nodo a visitar

            else:
                if(i < numVariables-1):
                    colonia[j][i+1] = proxNodo2(colonia[j][i], j)
            #############
            #Actualizar feromona local
            if(i < numVariables-1):
                matrizFeromona[colonia[j][i]][colonia[j][i+1]] = feromL(colonia[j][i], colonia[j][i+1])
    #Evaluar caminos encontrados para hallar mejores soluciones
    for i in range(col):
        costoCamino = solucionCalculaCosto(numVariables, colonia[i], matrizDistancias)
        if (costoCamino < solucionMejorCosto):            
            solucionMejor = colonia[i]
            solucionMejorCosto = costoCamino
            solucionMejorIteracion = generacion         
    #Actualizacion feromona global
    matrizFeromona = feromGlob(numVariables, solucionMejor, matrizFeromona, solucionMejorCosto)


    #print("Result Colonia \n",colonia, "\n")
    #print("Result Feromona \n",matrizFeromona, "\n") 

#################################################   

#Resultados
print('Resultados:')
##Calculo del tiempo que tomó el algoritmo
tiempo_proceso_fin = time.process_time()
print("Tiempo de procesamiento: %f segundos" %(tiempo_proceso_fin - tiempo_proceso_ini))
print('Mejor solución: ', solucionMejor)
print('Costo mejor solución: ', solucionMejorCosto)
print('Iteraciones hasta mejor solución: ', solucionMejorIteracion,'\n')

##Funcion que imprime grafo con las conexiones de las n variables
def imprimeGrafo(tam,sol):
    etiqueta = [x for x in range(tam)]
    lista = []
    for i in range(tam-1):
        par = []
        par.append(sol[i])
        par.append(sol[i+1])
        lista.append(par)
    lista.append([sol[tam-1],sol[0]])
    color = ['red'] * numVariables
    color[lista[0][0]] = 'blue'
    g = ig.Graph(n = tam, directed=True)
    g.add_edges(lista)
    g.vs["label"] = etiqueta
    g.vs["color"] = color
    g.vs["label_size"] = 6
    g.vs["size"] = 12
    g.es["edge_size"] = 2
    return g


mc=ig.Layout(coords=matrizCoordenadas.tolist())
ig.plot(imprimeGrafo(numVariables,solucionMejor), loyout=mc)



##plt.plot(imprimeGrafo(numVariables,solucionMejor), loyout=mc)

"""
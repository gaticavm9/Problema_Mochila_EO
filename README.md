# Trabajo-programacion-2
Desarrollo de una aplicación que implementa el Problema del Vendedor Viajero a través del método de Sistema de Colonia de Hormigas utilizando el lenguaje de programación Python
Se deben de setear los siguientes parámetros para la implementación del método:
-Valor semilla generador valores randómicos.
-Tamaño de la colonia o número de hormigas.
-Condición de término o número de iteraciones.
-Factor de evaporación de la feromona (α).
-El peso del valor de la heurística (β).
-Valor de probabilidad límite (q0).
-Archivo de entrada.


Al momento de ejecutar el codigo se deben pasar los parametros necesarios para el proceso.
El orden de los parametros a ingresar es: Semilla, TamañoColonia, NroIteraciones, TasaEvaportacion, Beta, q0, DatosEntrada

Ejemplo:
>> python berlin52.py 2 50 100 0.1 2.5 0.9 berlin52.tsp.txt





Resultados:
Tiempo de procesamiento: 52.171875 segundos
Mejor solución:  [29 22 19 49 28 15 45 43 33 34 35 38 39 37 36 47 23  4 14  5  3 24 11 27
 26 25 46 12 13 51 10 50 32 42  9  8  7 40 18 44 31 48  0 21 30 17  2 16
 20 41  6  1]
Costo mejor solución:  7548.992710024182
Iteraciones hasta mejor solución:  24
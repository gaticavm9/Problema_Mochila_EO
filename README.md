# Trabajo-programacion-3

Desarrollo de aplicación que implementa el *Problema de la Mochila* (Knapsack Problem) a través del método de *Extremal Optimization* utilizando el lenguaje de programación Python

## Installación
Descomprimir la aplicacion y abrir una terminal cambiando el directorio de trabajo actual a la ubicación en donde se encuantra la aplicación.
```bash
cd Trabajo-programacion-3
```

O clonar el repositorio a través del comando [git clone](https://docs.github.com/es/repositories/creating-and-managing-repositories/cloning-a-repository).

```bash
git clone https://github.com/victorex/Trabajo-programacion-3.git

cd Trabajo-programacion-3
```

## Importante
Al momento de ejecutar la aplicación se deben ingresar los siguientes parámetros necesarios para el funcionamiento de esta:
- Valor *Semilla* generador números randómicos.
- Valor de *Tau*.
- Número de iteraciones (condición de término).
- Nombre del archivo de entrada (archivo con datos para el análisis).

## Ejemplos de ejecución

```properties
 python mochila.py 1 1.4 1000 dat_3_200_1.txt
 python mochila.py 1 1.4 1000 dat_2_500_1000.txt
 python mochila.py 1 1.4 1000 dat_12_500_1000.txt
 python mochila.py 1 1.4 15000 dat_3_1000_100000.txt
```
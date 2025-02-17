import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd

class Habitacion:
    def __init__(self, filas, columnas, porcentaje_suciedad):
        self.filas = filas
        self.columnas = columnas
        self.matriz = np.zeros((filas, columnas), dtype=int)
        self.inicializar_suciedad(porcentaje_suciedad)
    
    def inicializar_suciedad(self, porcentaje_suciedad):
        total_celdas = self.filas * self.columnas
        celdas_sucias = int(total_celdas * (porcentaje_suciedad / 100))
        indices = random.sample(range(total_celdas), celdas_sucias)
        for idx in indices:
            fila, columna = divmod(idx, self.columnas)
            self.matriz[fila][columna] = 1  
    
    def esta_limpia(self):
        return np.sum(self.matriz) == 0

class RobotLimpieza:
    def __init__(self, habitacion, estrategia, x=0, y=0):
        self.habitacion = habitacion
        self.x = x
        self.y = y
        self.movimientos = 0
        self.estrategia = estrategia  
    
    def see(self):
        return self.habitacion.matriz[self.x][self.y] == 1
    
    def action(self):
        if self.see():
            self.habitacion.matriz[self.x][self.y] = 0  
    
    def next(self):
        if self.estrategia == "vertical":
            dx, dy = (random.choice([-1, 1]), 0)  
        else:
            dx, dy = (0, 0)  
        
        nuevo_x, nuevo_y = self.x + dx, self.y + dy
        if 0 <= nuevo_x < self.habitacion.filas and 0 <= nuevo_y < self.habitacion.columnas:
            self.x, self.y = nuevo_x, nuevo_y
            self.movimientos += 1


def simulacion(filas, columnas, k, porcentaje_suciedad, t_max, estrategia):
    habitacion = Habitacion(filas, columnas, porcentaje_suciedad)
    robots = [RobotLimpieza(habitacion, estrategia) for _ in range(k)]
    
    t = 0
    registros_corridas = []
    
    while not habitacion.esta_limpia() and t < t_max:
        for robot in robots:
            robot.action()
            robot.next()
        t += 1
        
        if t == int(0.25 * t_max) or t == int(0.5 * t_max) or t == int(0.75 * t_max) or t == t_max:
            registros_corridas.append((t, np.sum(habitacion.matriz)))
    
    porcentaje_limpieza = (1 - (np.sum(habitacion.matriz) / (filas * columnas))) * 100
    movimientos_totales = sum(robot.movimientos for robot in robots)
    
    return t, porcentaje_limpieza, movimientos_totales, registros_corridas

def ejecutar_experimentos(n_experimentos, filas, columnas, k, porcentaje_suciedad, t_max, estrategia):
    resultados = []
    with open("resultados_simulacion.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Experimento", "Estrategia", "Tiempo", "Porcentaje Limpieza", "Movimientos Totales"])
        
        for i in range(n_experimentos):
            tiempo, limpieza, movimientos, _ = simulacion(filas, columnas, k, porcentaje_suciedad, t_max, estrategia)
            writer.writerow([i+1, estrategia, tiempo, limpieza, movimientos])
            resultados.append((estrategia, tiempo, limpieza, movimientos))
    
    print(f"Resultados para estrategia {estrategia} guardados en resultados_simulacion.csv")

def graficar_resultados():
    df = pd.read_csv("resultados_simulacion.csv")
    plt.figure(figsize=(10, 5))
    
    plt.scatter(df["Tiempo"], df["Porcentaje Limpieza"], label="Vertical")
    plt.plot(df["Tiempo"], df["Porcentaje Limpieza"], linestyle='--')
    plt.xlabel("Tiempo")
    plt.ylabel("Porcentaje de Limpieza")
    plt.title("Tiempo vs Porcentaje de Limpieza (Vertical)")
    plt.legend()
    
    plt.scatter(df["Movimientos Totales"], df["Porcentaje Limpieza"], label="Vertical")
    plt.plot(df["Movimientos Totales"], df["Porcentaje Limpieza"], linestyle='--')
    plt.xlabel("Movimientos Totales")
    plt.ylabel("Porcentaje de Limpieza")
    plt.title("Movimientos vs Porcentaje de Limpieza (Vertical)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

estrategia = "vertical"
ejecutar_experimentos(5, 10, 10, 3, 30, 100, estrategia)
graficar_resultados()

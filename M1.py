import agentpy as ap
import numpy as np

class AgenteOptimo(ap.Agent):
    def setup(self):
        """ Configuración inicial del agente. """
        self.movimientos = 0  # Contador de movimientos

    def see(self):
        """ Observa el entorno y detecta si la celda está sucia. """
        return self.model.grid[self.position]  # Retorna True si la celda está sucia

    def next(self):
        """ Determina la próxima acción. """
        if self.see():
            self.model.grid[self.position] = False  # Limpia la celda
        else:
            dirty_cells = np.argwhere(self.model.grid)
            if len(dirty_cells) > 0:
                target = dirty_cells[np.argmin(np.linalg.norm(dirty_cells - self.position, axis=1))]
                self.position = tuple(target)  # Se mueve a la celda sucia más cercana
                self.movimientos += 1

class ModeloLimpieza(ap.Model):
    def setup(self):
        self.grid = np.random.choice([True, False], size=(10, 10), p=[0.3, 0.7])  # 30% de suciedad
        self.agentes = ap.AgentList(self, 5, AgenteOptimo)  # Crea 5 agentes
        for agente in self.agentes:
            agente.position = (0, 0)  # Todos empiezan en la celda (0,0)

    def step(self):
        self.agentes.next()

    def update(self):
        self.record('celdas_limpias', np.sum(~self.grid))

    def end(self):
        print(f"Porcentaje de celdas limpias: {np.sum(~self.grid) / self.grid.size * 100:.2f}%")
        print(f"Movimientos totales: {sum(agente.movimientos for agente in self.agentes)}")

# Ejecutar la simulación
modelo = ModeloLimpieza()
res = modelo.run(steps=50)

import agentpy as ap
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class cleaningAgent(ap.Agent):
    def setup(self):
      #inicializa a los agentes en la posicion [1,1]
        self.pos = (1,1)
        self.utilidad = 0
        self.moves=0
        self.time_step = 0
        self.percept = None
        self.cleaned_cells = set()
        self.visited_cells = set([self.pos])
        self.dirty_memo = set()
        

    #funcion para que el agente detecte si en su posición actual se encuentra
    #una celda sucia
    def see(self):
      #se obtiene la posicion actual del agente
      pos=self.model.grid.positions[self]
      #percibe las celdas sucias
      for dirty in self.model.dirt:
        if pos == self.model.grid.positions[dirty]:
          self.percept=dirty
          self.dirty_memo.add(pos)
        

    #funcion de toma de decisiones
    def next(self):
      #Decision 1: si la percepcion es de sucio, limpia y se mueve
      if self.percept:
        self.actions = [self.clean, self.move]
      #Desicion 2: caso contrario, solo se mueve
      else:
        self.actions = [self.move]
      pass

    #funcion para ejecutar las acciones
    def action(self):
      for act in self.actions:
        act()
      pass

#-----------------Acciones------------------------

    def move(self):
      #permite que el agente se mueva aleatoriamente dentro de las 8 direcciones
      #(izquierda,derecha,arriba,abajo,abajo-izquierda, abajo-derecha,
      #arriba-izquierda, arriba-derecha) mientras, respeta sus limites y
      #actualiza su posicion
      new_pos=((self.pos[0]+np.random.randint(-1,2))%self.model.p.limit_x,
       (self.pos[1]+np.random.randint(-1,2))%self.model.p.limit_y)
      tries=0
      self.pos = new_pos
      self.visited_cells.add(new_pos)
      self.moves += 1
     
      # si la celda ya fue visitada, intenta un par de veces para evitarla
      while new_pos in self.visited_cells and tries < 10:
        new_pos = ((self.pos[0] + np.random.randint(-1, 2)) % 
                   self.model.p.limit_x,
                    (self.pos[1] + np.random.randint(-1, 2)) 
                    % self.model.p.limit_y)
        tries += 1
      self.pos = new_pos
      self.visited_cells.add(new_pos)
      self.moves += 1

    def clean(self):
      #limpia las celdas sucias y, al estar limpias, las registra en memoria
      if self.percept in self.model.grid.positions:
        self.model.grid.remove_agents(self.percept)
        self.model.dirt.remove(self.percept)  # elimina la suciedad del modelo
        self.utilidad += 1 
        self.cleaned_cells.add(self.pos)  # registra la celda limpiada
        self.dirty_memo.discard(self.pos)  #quita la celda de la memo de suciedad
      self.percept = None 

class DirtyCell(ap.Agent):
  def setup(self):
    pass
  
class CleaningAgent(ap.Model):

    def setup(self):
      self.agents=ap.AgentList(self,self.p.agents,cleaningAgent)

      num_dirty_cells=int((self.p.limit_x*self.p.limit_y*self.p.dirty_cells)/100)
      self.dirt=ap.AgentList(self,num_dirty_cells,DirtyCell)

      self.grid=ap.Grid(self,[self.p.limit_x,self.p.limit_y],track_empty=True)

      self.grid.add_agents(self.agents,random=True)
      self.grid.add_agents(self.dirt,random=True)

      self.total_steps=0

    def step(self):
        self.agents.see()
        self.agents.next()
        self.agents.action()
        self.total_steps+=1

        #comprueba si ya se han limpiado todas las celdas
        #si es asi, termina el programa
        if len(self.dirt)==0:
          self.end()
          self.stop()
        #termina el programa si se acabo el tiempo
        elif self.total_steps>=self.p.max_steps:
          self.end()
          self.stop()

    def update(self):
        self.record('total_steps',self.total_steps)

    def end(self):
      remainig_dirty_cells= len(self.dirt)
      total_cells = self.p.limit_x * self.p.limit_y
      clean_cells_percentage=(total_cells-remainig_dirty_cells)/total_cells*100
      utilidad = sum(a.utilidad for a in self.agents)
      movements=sum(a.moves for a in self.agents)

      self.record('utilidad',utilidad)
      self.record('moves',movements)
      self.record('remaining_dirty_cells', remainig_dirty_cells)
      self.record('clean_cells_percentage', clean_cells_percentage)
      self.record('total_steps', self.total_steps)


#funcion para recopilar resultados y ejecutar simulacion
def run(percentage):
  parameters={
    'limit_x':10,
    'limit_y':8,
    'agents':10,
    'max_steps':int(222*percentage),
    'dirty_cells':45
    }
  return parameters

simula = []
for percentage in [0.25, 0.50, 0.75, 1.0]:
  model = CleaningAgent(run(percentage))
  results = model.run()
  cleaning_agent_df = results.variables['CleaningAgent']
  # Obtiene  los últimos valores de cada variable y los junta
  simula.append({
      'run_time_percentage': percentage * 100,
      'remaining_dirty_cells':cleaning_agent_df['remaining_dirty_cells'].iloc[-1],
      'clean_cells_percentage':cleaning_agent_df['clean_cells_percentage'].iloc[-1],
      'utilidad': cleaning_agent_df['utilidad'].iloc[-1],
      'total_steps': cleaning_agent_df['total_steps'].iloc[-1],
      'moves': cleaning_agent_df['moves'].iloc[-1]
    })

df = pd.DataFrame(simula)

# Crear subgráficos
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Gráfico 1: Porcentaje de celdas limpias vs. Tiempo
axes[0, 0].plot(df['run_time_percentage'], df['clean_cells_percentage'],
                marker='o', linestyle='-', color='b')
axes[0, 0].set_title("Porcentaje de Celdas Limpias vs. Tiempo")
axes[0, 0].set_xlabel("Tiempo de Ejecución (%)")
axes[0, 0].set_ylabel("Celdas Limpias (%)")

# Gráfico 2: Número de movimientos vs. Tiempo
axes[0, 1].bar(df['run_time_percentage'], df['moves'], color='g')
axes[0, 1].set_title("Número de Movimientos vs. Tiempo")
axes[0, 1].set_xlabel("Tiempo de Ejecución (%)")
axes[0, 1].set_ylabel("Movimientos")

# Gráfico 3: Utilidad obtenida vs. Tiempo
axes[1, 0].plot(df['run_time_percentage'], df['utilidad'], marker='s',
                linestyle='--', color='r')
axes[1, 0].set_title("Utilidad Obtenida vs. Tiempo")
axes[1, 0].set_xlabel("Tiempo de Ejecución (%)")
axes[1, 0].set_ylabel("Utilidad")

# Gráfico 4: Celdas sucias restantes vs. Tiempo
axes[1, 1].bar(df['run_time_percentage'], df['remaining_dirty_cells'],
               color='orange')
axes[1, 1].set_title("Celdas Sucias Restantes vs. Tiempo")
axes[1, 1].set_xlabel("Tiempo de Ejecución (%)")
axes[1, 1].set_ylabel("Celdas Sucias Restantes")

# Ajustar diseño y mostrar
plt.tight_layout()
plt.show()

def calcular_puntuacion(agent_data, max_steps, total_cells):
    utilidad = agent_data['utilidad']
    clean_cells_percentage = agent_data['clean_cells_percentage']
    total_steps = agent_data['total_steps']
    remaining_dirty_cells = agent_data['remaining_dirty_cells']
    moves = agent_data['moves']
    
    # se definen las ponderaciones
    peso_utilidad = 0.3
    peso_limpieza = 0.3
    peso_steps = 0.1
    peso_suciedad = 0.2
    peso_movimientos = 0.1

    # encuentra el agente óptimo al comparar las puntuaciones
    max_utilidad = max([s['utilidad'] for s in simula])
    max_moves = max([s['moves'] for s in simula]) 
    
    # calcula la puntuación para cada métrica
    puntuacion = (
        (utilidad/max_utilidad)*peso_utilidad +
        (clean_cells_percentage/100)*peso_limpieza +
        ((1 - (total_steps / max_steps)) * peso_steps) +
        ((1 - (total_cells - remaining_dirty_cells / total_cells)) * 
         peso_suciedad) +
        (((max_moves - moves) / max_moves) * peso_movimientos)
    )
    
    return puntuacion

simula_puntuadas = []
max_steps = max([s['total_steps'] for s in simula])
total_cells = 8*10

# calcula la puntuación para cada simulación
for agent_data in simula:
    puntuacion = calcular_puntuacion(agent_data, max_steps, total_cells)
    agent_data['puntuacion'] = puntuacion
    simula_puntuadas.append(agent_data)

# cordena por la puntuación
simula_puntuadas.sort(key=lambda x: x['puntuacion'], reverse=True)

agente_optimo = simula_puntuadas[0]
print(f"El agente óptimo tiene una puntuación de {agente_optimo['puntuacion']:.4f}")
print(f"Detalles del agente óptimo: {agente_optimo}")


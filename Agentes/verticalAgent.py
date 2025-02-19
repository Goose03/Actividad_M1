import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np

class VacuumSnakeAgent(ap.Agent):
    def setup(self):
        self.cleaned = 0         
        self.path_history = []   
        self.done = False        

    def see(self):
        current_pos = self.model.grid.positions[self]
        return self.model.grid.clean[current_pos]

    def next(self):
        x, y = self.model.grid.positions[self]
        max_x = self.model.p.x - 1
        max_y = self.model.p.y - 1
        if x % 2 == 0:
            if y < max_y:
                return (x, y + 1)
            elif x < max_x:
                return (x + 1, y)
            else:
                return None        
        else:
            if y > 0:
                return (x, y - 1)
            elif x < max_x:
                return (x + 1, y)
            else:
                return None

    def action(self):
        current_pos = self.model.grid.positions[self]
        x, y = current_pos   
        if self.see():
            self.model.grid.clean[current_pos] = False
            self.cleaned += 1
            self.path_history.append(f"limpiado ({x},{y})")
        else:
            self.path_history.append(f"({x},{y})")
        next_pos = self.next()
        if next_pos is not None:
            self.model.grid.move_to(self, next_pos)
        else:
            self.done = True

class VacuumSnakeModel(ap.Model):
    def setup(self):
        self.grid = ap.Grid(self, (self.p.x, self.p.y), track_empty=True)
        self.agents = ap.AgentList(self, self.p.agents, VacuumSnakeAgent)
        start_pos = (1, 1)

        self.grid.add_agents(self.agents, positions=[start_pos] * self.p.agents)
        self.grid.add_field('clean', values=np.full(self.grid.shape, False))

        total_cells = self.p.x * self.p.y
        num_dirty = int(total_cells * self.p.dirt_percentage)
        dirty_positions = np.random.choice(total_cells, num_dirty, replace=False)
        xs, ys = np.unravel_index(dirty_positions, self.grid.shape)

        for x, y in zip(xs, ys):
            self.grid.clean[(x, y)] = True

        self.dirty_remaining = num_dirty
        self.cleaning_progress = []
        self.steps_taken = 0

    def step(self):

        for agent in self.agents:
            if not agent.done:
                cleaned_before = agent.cleaned
                agent.action()
                cleaned_after = agent.cleaned
                if cleaned_after > cleaned_before:
                    self.dirty_remaining -= (cleaned_after - cleaned_before)

        cleaned_total = (self.p.x * self.p.y) - self.dirty_remaining
        self.cleaning_progress.append(cleaned_total)
        self.steps_taken += 1

    def update(self):

        if self.dirty_remaining == 0:
            self.stop()

        if all(a.done for a in self.agents):
            self.stop()

    def end(self):
        total_cleaned = (self.p.x * self.p.y) - self.dirty_remaining
        print(f"\nla simulación terminó en {self.steps_taken} pasos.")
        print(f"celdas limpiadas: {total_cleaned} / {self.p.x * self.p.y}")
        print(f"porcentaje limpiado: {100 * total_cleaned / (self.p.x * self.p.y):.2f}%")
        return self.cleaning_progress

def RunSimulation():

    max_steps = 100
    fractions = [0.25, 0.50, 0.75, 1.00]
    labels = ['A', 'B', 'C', 'D']
    runs = {}
    
    for frac, label in zip(fractions, labels):
        p = {
            'x': 10,
            'y': 10,
            'agents': 1,
            'dirt_percentage': 0.3
        }
        model = VacuumSnakeModel(p)
        model.setup()
        model.running = True
        step_limit = int(max_steps * frac)
        for _ in range(step_limit):
            if not model.running:
                break
            model.step()
            model.update()
        final_progress = model.end()
        runs[label] = final_progress
    plt.figure(figsize=(6, 4))

    for label, progress in runs.items():
        plt.plot(progress, label=f'Run {label}')

    plt.xlabel('Time Step')
    plt.ylabel('Cells Cleaned')
    plt.title('Vacuum Snake Cleaning Progress')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    RunSimulation()
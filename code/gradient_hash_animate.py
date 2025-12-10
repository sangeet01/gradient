import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import hashlib
import random

class Server:
    def __init__(self, server_id, x, y, capacity=2000):
        self.id = server_id
        self.capacity = capacity
        # For animation, we track "recent" load to show pulsation
        self.current_load = 0
        self.x = x
        self.y = y
        self.alive = True

    def get_pheromone(self):
        if not self.alive: return 0.0001
        load_ratio = self.current_load / self.capacity
        if load_ratio > 1.0: load_ratio = 1.0
        return load_ratio

class ACHHasher:
    def __init__(self, servers):
        self.servers = servers
    
    def get_server(self, key, sx, sy):
        best_server = None
        max_flow = -1.0
        
        for server in self.servers:
            # Base affinity
            pair_key = f"{key}-{server.id}"
            base_affinity = (int(hashlib.md5(pair_key.encode()).hexdigest(), 16) % 1000) / 1000.0
            
            dist_sq = (server.x - sx)**2 + (server.y - sy)**2
            proximity_signal = 1.0 / (dist_sq + 0.01)
            
            # Pheromone
            # If dead using 0.0001 capacity -> Huge pressure -> 0 flow
            load_ratio = server.current_load / server.capacity
            capacity_signal = 1.0 / (1.0 + (load_ratio * 5)**2)
            
            noise = 0.9 + (base_affinity * 0.2)
            
            flow = proximity_signal * capacity_signal * noise
            
            if flow > max_flow:
                max_flow = flow
                best_server = server
        return best_server

# Setup
coords = [(0.2,0.2), (0.8,0.2), (0.5,0.5), (0.2,0.8), (0.8,0.8)]
servers = [Server(f"S{i}", x, y) for i, (x,y) in enumerate(coords)]
ach = ACHHasher(servers)

# We want to animate a stream of requests over time
# We specifically target S0 (Bottom Left: 0.2, 0.2)
target_x, target_y = 0.2, 0.2

fig, ax = plt.subplots(figsize=(6,6))

def update(frame):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"ACH Traffic Flow (Frame {frame})\nS0 Targeted")
    
    # Logic:
    # Frame 0-30: Normal
    # Frame 30: Kill S0
    if frame == 30:
        servers[0].alive = False
        servers[0].capacity = 0.01 # Virtually dead
        
    # Simulate a batch of 50 requests per frame
    batch_points = {s.id: [] for s in servers}
    
    # Decay load slightly each frame to simulate processing
    for s in servers:
        s.current_load *= 0.9
        
    # Generate new traffic
    for i in range(50):
        key = f"anim-{frame}-{i}"
        # Traffic centered on S0
        rx = target_x + random.uniform(-0.15, 0.15)
        ry = target_y + random.uniform(-0.15, 0.15)
        
        s = ach.get_server(key, rx, ry)
        s.current_load += 1
        batch_points[s.id].append((rx, ry))

    # Plot Servers
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, s in enumerate(servers):
        # Color logic: Gray if dead
        c = 'gray' if not s.alive else colors[i]
        # Size logic: Pulse with load
        size = 200 + (s.current_load * 2) 
        ax.scatter(s.x, s.y, c=c, s=size, marker='s', edgecolors='black', zorder=10)
        ax.text(s.x, s.y+0.05, s.id, ha='center')

    # Plot Traffic Dots
    # We aggregate dots for visualization
    for i, s in enumerate(servers):
        pts = batch_points[s.id]
        if pts:
            px, py = zip(*pts)
            # If server is dead, these are "lost" packets (shouldn't happen in ACH)
            # If alive, color matches server
            c = colors[i]
            ax.scatter(px, py, c=c, alpha=0.6, s=20)

ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
ani.save('ach_resilience.gif', writer='pillow')
print("Saved ach_resilience.gif")

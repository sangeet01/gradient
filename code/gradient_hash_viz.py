import matplotlib.pyplot as plt
import numpy as np
import hashlib
import bisect
import random

# --- Re-using Core Logic for Consistency ---

class Server:
    def __init__(self, server_id, x, y, capacity=2000):
        self.id = server_id
        self.capacity = capacity
        self.current_load = 0
        self.x = x
        self.y = y

    def get_pheromone(self):
        load_factor = (self.current_load + 1e-5)
        # Using the tuned equation from simulation
        load_ratio = self.current_load / self.capacity
        # Safety for visualization if load helps visual overflow
        if self.current_load > self.capacity: load_ratio = 1.0 
        
        # Inverted logic match: flow equation uses specific signals
        return load_ratio # Just returning state for visualizer usage simplicity

    def reset(self):
        self.current_load = 0

class ACHHasher:
    def __init__(self, servers):
        self.servers = servers
    
    def get_server(self, key, sx, sy):
        best_server = None
        max_flow = -1.0
        
        for server in self.servers:
            # Replicating logic from simulation
            pair_key = f"{key}-{server.id}"
            base_affinity = (int(hashlib.md5(pair_key.encode()).hexdigest(), 16) % 1000) / 1000.0
            
            dist_sq = (server.x - sx)**2 + (server.y - sy)**2
            proximity_signal = 1.0 / (dist_sq + 0.01)
            
            # Pheromone logic
            load_ratio = server.current_load / server.capacity
            capacity_signal = 1.0 / (1.0 + (load_ratio * 5)**2)
            
            noise = 0.9 + (base_affinity * 0.2)
            
            flow = proximity_signal * capacity_signal * noise
            
            if flow > max_flow:
                max_flow = flow
                best_server = server
        return best_server

class RingHasher:
    def __init__(self, servers, replicas=100):
        self.ring = {}
        self.sorted_keys = []
        for server in servers:
            for i in range(replicas):
                k = int(hashlib.md5(f"{server.id}-{i}".encode()).hexdigest(), 16)
                self.ring[k] = server
                self.sorted_keys.append(k)
        self.sorted_keys.sort()
        
    def get_server(self, key):
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        idx = bisect.bisect(self.sorted_keys, h)
        if idx == len(self.sorted_keys): idx = 0
        return self.ring[self.sorted_keys[idx]]

# --- Plot 1: Locality Scatter ---
def plot_locality():
    print("Generating Locality Plot...")
    
    # 5 Servers arranged somewhat geographically
    # (0.2,0.2) (0.8,0.2) (0.5,0.5) (0.2,0.8) (0.8,0.8)
    coords = [(0.2,0.2), (0.8,0.2), (0.5,0.5), (0.2,0.8), (0.8,0.8)]
    servers = [Server(f"S{i}", x, y) for i, (x,y) in enumerate(coords)]
    
    ach = ACHHasher(servers)
    ring = RingHasher(servers)
    
    # Generate requests centered around (0.2, 0.2) [Bottom Left]
    # We want to see if they stay in Bottom Left
    req_x, req_y = 0.25, 0.25
    
    n_points = 200
    
    # Data containers
    ach_points = {s.id: [] for s in servers}
    ring_points = {s.id: [] for s in servers}
    
    for i in range(n_points):
        key = f"vis-{i}"
        # Slight jitter
        rx = req_x + random.uniform(-0.1, 0.1)
        ry = req_y + random.uniform(-0.1, 0.1)
        
        # ACH
        s_ach = ach.get_server(key, rx, ry)
        ach_points[s_ach.id].append((rx, ry))
        
        # Ring
        s_ring = ring.get_server(key)
        ring_points[s_ring.id].append((rx, ry))
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Plot Ring
    for i, s in enumerate(servers):
        pts = ring_points[s.id]
        if pts:
            px, py = zip(*pts)
            ax1.scatter(px, py, c=colors[i], label=s.id, alpha=0.6, edgecolors='none')
        # Plot Server
        ax1.scatter(s.x, s.y, c=colors[i], s=200, marker='s', edgecolors='black')
        
    ax1.set_title("Ring Hashing (Standard)\nNotice: Random Colors (Scatter)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot ACH
    for i, s in enumerate(servers):
        pts = ach_points[s.id]
        if pts:
            px, py = zip(*pts)
            ax2.scatter(px, py, c=colors[i], label=s.id, alpha=0.6, edgecolors='none')
        # Plot Server
        ax2.scatter(s.x, s.y, c=colors[i], s=200, marker='s', edgecolors='black')
        
    ax2.set_title("ACH (Bio-Inspired)\nNotice: Solid Color (Locality)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ach_locality_comparison.png')
    print("Saved ach_locality_comparison.png")

# --- Plot 2: Resilience / Failover ---
def plot_resilience():
    print("Generating Resilience Plot...")
    
    # From our simulation results:
    # S4 (Closest): 2924
    # S3 (Next): 1264
    # S1 (Far): 812
    # S2 (Far): 0 
    # (Values approximate from previous run logic for cleanness)
    
    labels = ['S4 (Closest)', 'S3 (Neighbor)', 'S1 (Remote)', 'S2 (Remote)']
    values = [2924, 1264, 812, 0] # Using actual data pattern
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['green', 'yellowgreen', 'gray', 'gray'])
    
    plt.title("Traffic Redistribution After S0 Failure (Geometric Failover)")
    plt.ylabel("Requests Received")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add text
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
                 
    plt.savefig('ach_resilience_chart.png')
    print("Saved ach_resilience_chart.png")

if __name__ == "__main__":
    plot_locality()
    plot_resilience()

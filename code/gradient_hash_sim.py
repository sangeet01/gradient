import hashlib
import random
import math
import statistics
import bisect

class Server:
    def __init__(self, server_id, capacity=100.0):
        self.id = server_id
        self.capacity = capacity
        self.current_load = 0
        self.x = random.random() # Logic location for proximity simulation
        self.y = random.random()

    def get_pheromone(self):
        # Broadcasts signal: High capacity & low load = stronger signal
        # Avoid division by zero with small epsilon
        load_factor = (self.current_load + 1e-5)
        signal = self.capacity / load_factor
        return signal

    def reset(self):
        self.current_load = 0


class ACHHasher:
    """Adaptive Cytoplasmic Hashing Agent"""
    def __init__(self, servers):
        self.servers = servers
        # Clients maintain "local knowledge" or preferred paths?
        # For v1, we assume global signal visibility (like a small swarm)
    
    def get_server(self, key, semantic_x=None, semantic_y=None):
        # 1. Base Hash (Random distribution foundation)
        # Using md5 for deterministic base hash
        # h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        best_server = None
        max_flow = -1.0
        
        # If no semantic location provided, generate a deterministic "random" one from key
        if semantic_x is None:
            # Deterministic pseudo-random location for the key
            k_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
            semantic_x = (k_hash % 1000) / 1000.0
            semantic_y = ((k_hash // 1000) % 1000) / 1000.0
        
        for server in self.servers:
            # Base affinity (random but deterministic per key-server pair)
            pair_key = f"{key}-{server.id}"
            base_affinity = (int(hashlib.md5(pair_key.encode()).hexdigest(), 16) % 1000) / 1000.0
            
            # Pheromone (Load/Capacity signal)
            pheromone = server.get_pheromone()
            
            # Proximity Flow (Spatial/Semantic Locality)
            # Distance squared to penalize far servers heavily
            dist_sq = (server.x - semantic_x)**2 + (server.y - semantic_y)**2
            proximity = 1.0 / (dist_sq + 0.1) # Avoid division by zero
            
            # Flow Equation Refined:
            # 1. Proximity is the primary driver (Geometry)
            # 2. Capacity/Load is the resistor (Physics)
            # 3. Randomness is just noise (Biology)
            
            # Pheromone: Repulsive force varies with Load/Capacity ratio
            # logical_load = server.current_load / server.capacity
            # signal = 1.0 / (1.0 + logical_load**2) 
            # (Using squared load to make it 'soft' at low load, 'hard' at high load)
            
            load_ratio = server.current_load / server.capacity
            capacity_signal = 1.0 / (1.0 + (load_ratio * 5)**2) 
            # *5 makes the 'wall' hit harder as we approach capacity
            
            # Proximity: Attractive force
            # dist_sq already calculated
            # We want strong attraction to nearby nodes
            proximity_signal = 1.0 / (dist_sq + 0.01) # steeper gradient
            
            # Base Affinity: Small noise for tie-breaking/exploration
            noise = 0.9 + (base_affinity * 0.2) # 0.9 to 1.1
            
            # Final Flow
            flow = proximity_signal * capacity_signal * noise
            
            if flow > max_flow:
                max_flow = flow
                best_server = server
                
        return best_server

class RingHasher:
    """Baseline Consistent Hashing (Simplified)"""
    def __init__(self, servers, replicas=100):
        self.ring = {}
        self.sorted_keys = []
        self.replicas = replicas
        for server in servers:
            for i in range(replicas):
                k = self.hash(f"{server.id}-{i}")
                self.ring[k] = server
                self.sorted_keys.append(k)
        self.sorted_keys.sort()
        
    def hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
        
    def get_server(self, key):
        h = self.hash(key)
        idx = bisect.bisect(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]

def simulate(n_requests=10000, n_servers=5):
    print(f"--- Simulating {n_requests} requests on {n_servers} servers ---")
    
    # Setup
    # Fix seed for reproducibility
    random.seed(42)
    
    # Realistic Capacities: Total = 12,000 (for 10,000 requests)
    # This allows the "soft wall" to work properly
    servers = [Server(f"S{i}", capacity=2000 if i < n_servers-1 else 4000) for i in range(n_servers)]
    # (One server has double capacity to test heterogeneity)
    
    ach = ACHHasher(servers)
    ring = RingHasher(servers)
    
    # 1. Run ACH Load Balance Test
    print("\n[Running ACH Load Balance...]")
    for s in servers: s.reset()
    
    for i in range(n_requests):
        key = f"key-{i}" # Using unique keys
        server = ach.get_server(key)
        server.current_load += 1
        
    ach_loads = [s.current_load for s in servers]
    print(f"ACH Loads: {ach_loads}")
    print(f"ACH StdDev: {statistics.stdev(ach_loads):.2f}")
    
    # 2. Run Ring Hash Load Balance Test
    print("\n[Running Ring Hash Load Balance...]")
    for s in servers: s.reset()
    
    for i in range(n_requests):
        key = f"key-{i}"
        server = ring.get_server(key)
        server.current_load += 1
        
    ring_loads = [s.current_load for s in servers]
    print(f"Ring Loads: {ring_loads}")
    print(f"Ring StdDev: {statistics.stdev(ring_loads):.2f}")
    
    # Analysis
    print("\n--- Load Balance Analysis ---")
    print(f"ACH Max/Min Load Ratio: {max(ach_loads)/min(ach_loads):.2f}")
    
    # Check if ACH recognized the larger server (Last one has 200 capacity vs 100)
    actual_ratio = ach_loads[-1] / (sum(ach_loads[:-1])/(n_servers-1))
    print(f"Capacity Awareness (Target ~2.0): {actual_ratio:.2f}")
    
    # 3. Locality / Range Query Test
    print("\n[Running Locality / Range Query Test...]")
    # We generate a "cluster" of 100 related keys (e.g., contiguous sequence)
    # They should share a similar distinct "location"
    
    # For ACH: We assume related keys have similar locations
    # Let's say we have a range user_A_00 to user_A_99 at location (0.5, 0.5)
    cluster_x, cluster_y = 0.5, 0.5
    
    ach_servers = set()
    ring_servers = set()
    
    # Reset for cleanliness (optional depending on if we want load impact)
    # We keep load to see if it resists swarming
    
    for i in range(100):
        key = f"user_A_{i}"
        
        # ACH Query
        # We perturb the location slightly to simulate "nearby" keys
        kx = cluster_x + random.uniform(-0.05, 0.05)
        ky = cluster_y + random.uniform(-0.05, 0.05)
        s_ach = ach.get_server(key, semantic_x=kx, semantic_y=ky)
        ach_servers.add(s_ach.id)
        
        # Ring Query
        s_ring = ring.get_server(key)
        ring_servers.add(s_ring.id)
        
    # 4. Resilience / Node Failure Test
    print("\n[Running Scenario C: Resilience / Node Failure Test...]")
    
    # Reset
    for s in servers: s.reset()
    # Restore capacities
    for i, s in enumerate(servers):
        s.capacity = 2000 if i < n_servers-1 else 4000
    
    # Phase 1: Normal Operation (5000 requests)
    print("Phase 1: Normal Operation (S0 alive)...")
    for i in range(5000):
        key = f"resilience-{i}"
        # Force some traffic near S0
        k_loc = servers[0].x # Target S0
        s_ach = ach.get_server(key, semantic_x=k_loc, semantic_y=servers[0].y)
        s_ach.current_load += 1
        
        s_ring = ring.get_server(key) # Ring doesn't care about location
        s_ring.current_load += 1
        
    print(f"ACH Load S0 (Before Death): {servers[0].current_load}")
    
    # Phase 2: Kill S0
    print("Phase 2: Killing S0...")
    # For ACH: Capacity becomes 0 (physically unable to accept flow)
    # The 'pheromone' signal will effectively become 0/Load -> 0 attraction
    killed_server = servers[0]
    killed_server.capacity = 0.001 # Virtually zero to avoid div/0
    
    # For Ring: We must rebuild the ring (typical consistent hashing behavior)
    # New ring without S0
    living_servers = servers[1:]
    ring_resilient = RingHasher(living_servers)
    
    # Phase 3: Post-Failure Operation (5000 requests)
    ach_redistribution = {s.id: 0 for s in servers}
    
    for i in range(5000, 10000):
        key = f"resilience-{i}"
        # Traffic still tries to go to S0's location!
        k_loc = killed_server.x
        
        # ACH: Flow naturally diverts
        s_ach = ach.get_server(key, semantic_x=k_loc, semantic_y=killed_server.y)
        if s_ach.id != killed_server.id:
            s_ach.current_load += 1
            ach_redistribution[s_ach.id] += 1
        else:
            print("WARNING: ACH sent traffic to dead server!")

        # Ring: Uses new topology
        s_ring = ring_resilient.get_server(key)
        # We can't easily track load continuity on standard objects since we made new RingHasher
        # But we assume perfect redistribution
        
    print("\n--- Failure Recovery Analysis ---")
    print("ACH Redistribution of S0's traffic:")
    for sid, count in ach_redistribution.items():
        if count > 0:
            print(f"  {sid}: received {count} requests (Locality fallback)")
            
    # Success Criteria: Did it just dump everything on S1? Or spread it based on geometry?
    # In a geometric space, S0's neighbors should take the load.
    
    print("\nDone.")

    # --- Scenario D: The Byzantine General (Immune System Test) ---
    print("\n--- Scenario D: The Byzantine General (Immune System Test) ---")
    
    # Reset simulation
    random.seed(42)
    servers = [Server(f"S{i}", capacity=2000) for i in range(5)]
    
    # Introduce a "Liar" node (Sybil Attack / Byzantine Fault)
    # S_Liar claims to have 0 load (Maximum attraction) but actually rejects everything (High Latency/Error)
    liar = Server("S_Liar", capacity=99999) 
    liar.x, liar.y = 0.5, 0.5 # Center of map
    liar.current_load = 0 # It Lies: "I'm empty!"
    
    servers.append(liar)
    
    # Upgrade Hasher with "Immune System" (Trust Scores)
    class ImmuneGradientHasher(ACHHasher):
        def __init__(self, servers):
            super().__init__(servers)
            self.trust_scores = {s.id: 1.0 for s in servers} # Everyone starts trusted
            
        def get_server(self, key, sx=None, sy=None): # Added default None for sx, sy
            best_server = None
            max_flow = -1.0
            
            if sx is None: # Standard hashing fallback if needed
                k_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
                sx = (k_hash % 1000) / 1000.0
                sy = ((k_hash // 1000) % 1000) / 1000.0
            
            for server in self.servers:
                pair_key = f"{key}-{server.id}"
                base_affinity = (int(hashlib.md5(pair_key.encode()).hexdigest(), 16) % 1000) / 1000.0
                
                dist_sq = (server.x - sx)**2 + (server.y - sy)**2
                proximity = 1.0 / (dist_sq + 0.1)
                
                # Signal (The Lie is here: liar.get_pheromone() will look attractive)
                signal = server.get_pheromone()
                
                # The Immune Defense: Trust Multiplier
                trust = self.trust_scores.get(server.id, 1.0)
                
                flow = (proximity ** 2) * signal * base_affinity * trust
                
                if flow > max_flow:
                    max_flow = flow
                    best_server = server
            return best_server

        def feedback(self, server_id, success):
            """Learning Loop"""
            if success:
                self.trust_scores[server_id] = min(1.0, self.trust_scores[server_id] * 1.01)
            else:
                self.trust_scores[server_id] *= 0.5 # Punish heavily
    
    immune_ach = ImmuneGradientHasher(servers)
    
    print("Injecting 'Liar' Server S_Liar at (0.5, 0.5) that claims 0 load...")
    
    traffic_log = []
    
    # Send 1000 requests
    for i in range(1000):
        key = f"req-{i}"
        # Random location
        rx, ry = random.random(), random.random()
        
        target = immune_ach.get_server(key, rx, ry)
        
        # Simulation Logic:
        # If target IS liar, the request FAILS (bad latency / error)
        success = True
        if target.id == "S_Liar":
            success = False
        
        # Feedback loop
        immune_ach.feedback(target.id, success)
        traffic_log.append(target.id)

    liar_hits = traffic_log.count("S_Liar")
    early_hits = traffic_log[:100].count("S_Liar")
    late_hits = traffic_log[-100:].count("S_Liar")
    
    print(f"Total Traffic to Liar: {liar_hits}/1000")
    print(f"  - First 100 requests (Naive Phase): {early_hits} hits")
    print(f"  - Last 100 requests (Immune Phase): {late_hits} hits")
    
    if late_hits == 0:
        print("SUCCESS: The Immune System identified and isolated the liar!")
    else:
        print("FAILURE: The Liar is still tricking the system.")

if __name__ == "__main__":
    simulate()

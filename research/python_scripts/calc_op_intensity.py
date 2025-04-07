"""
Operational Intensity Calculator for PET Scheduling (Algorithm 2)

Implements the computation of operational intensities as described in the paper:
- Input: PET operators and streams
- Uses latency model β (beta_table) 
- Derives bandwidth utilization from execution data
- Output: Operator to stream assignment with intensities
"""

class OperationalIntensityCalculator:
    def __init__(self, beta_model_path):
        """Initialize with beta latency model"""
        self.beta_model = self.load_beta_model(beta_model_path)
        self.stream_capacity = 1e9  # Default stream bandwidth (1GB/s)
        
    def load_beta_model(self, path):
        """Load beta latency model data"""
        model = {}
        with open(path, 'r') as f:
            for line in f:
                pet_type, bs, seq_len, latency = map(float, line.strip().split())
                model[(int(pet_type), int(bs), int(seq_len))] = latency
        return model
    
    def calculate_flops(self, pet_type, batch_size, seq_len):
        """Calculate FLOPs for PET operator"""
        hidden_size = 768  # BERT hidden size
        
        if pet_type == 0:
            return 2 * batch_size * seq_len * hidden_size**2
        elif pet_type == 3:
            return 8 * batch_size * seq_len**2 * hidden_size
        else:
            raise ValueError(f"Unsupported PET type: {pet_type}")
    
    def estimate_bandwidth(self, pet_type, batch_size, seq_len):
        """Estimate bandwidth usage from latency model"""
        key = (pet_type, batch_size, seq_len)
        if key not in self.beta_model:
            raise ValueError(f"No beta model data for {key}")
            
        latency_ms = self.beta_model[key]
        # Approximate bytes accessed based on latency
        return (batch_size * seq_len * 768 * 4) / (latency_ms / 1000)  # Simplified model
    
    def compute_intensities(self, operators):
        """Compute operational intensities for PET operators"""
        intensities = []
        for op in operators:
            pet_type, bs, seq_len = op
            flops = self.calculate_flops(pet_type, bs, seq_len)
            bandwidth = self.estimate_bandwidth(pet_type, bs, seq_len)
            intensity = flops / bandwidth
            intensities.append((op, intensity))
        return intensities
    
    def schedule_operators(self, operators, streams):
        """Implement scheduling Algorithm 2 from paper"""
        # Step 1: Compute operational intensities
        op_intensities = self.compute_intensities(operators)
        
        # Sort operators by intensity (high to low)
        sorted_ops = sorted(op_intensities, key=lambda x: -x[1])
        
        # Simple round-robin assignment to streams
        assignments = {}
        for i, (op, intensity) in enumerate(sorted_ops):
            stream_idx = i % len(streams)
            assignments[op] = (stream_idx, intensity)
            
        return assignments

# Example usage
if __name__ == "__main__":
    # Initialize with beta model
    calculator = OperationalIntensityCalculator("research/python_scripts/perf_model/beta_table_1080ti.dat")
    
    # Example PET operators to schedule (pet_type, batch_size, seq_len)
    operators = [
        (0, 32, 128),
        (0, 16, 64), 
        (3, 8, 32),
        (3, 4, 16)
    ]
    
    # Available streams
    streams = [0, 1, 2]  # 3 streams
    
    # Run scheduling algorithm
    assignments = calculator.schedule_operators(operators, streams)
    
    # Print results
    print("PET Operator Scheduling Assignments:")
    print("Operator (type, bs, seq) -> (stream, intensity)")
    for op, (stream, intensity) in assignments.items():
        print(f"{op} -> ({stream}, {intensity:.2f})")
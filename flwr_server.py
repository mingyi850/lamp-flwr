import flwr as fl
import time 
from memory_profiler import profile

@profile
def main():
    start_time = time.time()
    num_rounds = 100
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=3))
    total_time_taken = time.time() - start_time
    print("Total time taken: ", total_time_taken)

if __name__ == "__main__":
    main()
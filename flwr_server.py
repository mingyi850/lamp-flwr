import flwr as fl
import time 
import os 

def main():
    num_rounds = 100
    master_addr = os.getenv('MASTER_ADDR', '0.0.0.0')
    master_port = os.getenv('MASTER_PORT', '8080')
    print("Serving on address:", f"{master_addr}:{master_port}")
    start_time = time.time()
    fl.server.start_server(server_address=f"{master_addr}:{master_port}", config=fl.server.ServerConfig(num_rounds=100))
    total_time_taken = time.time() - start_time
    print("Total time taken: ", total_time_taken)

if __name__ == "__main__":
    main()
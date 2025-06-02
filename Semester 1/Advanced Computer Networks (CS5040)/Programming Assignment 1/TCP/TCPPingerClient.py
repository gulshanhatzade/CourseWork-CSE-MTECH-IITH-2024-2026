import time
import socket

server_ip_address = ('192.168.0.222', 11003)  

def tcp_pinger_client(N):
    client_socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Creating a TCP socket 
    
    client_socket_tcp.connect(server_ip_address) # Connecting to the server

    client_socket_tcp.settimeout(1.0) # 1 second timeout 

    rtt_times = []         # RTT statistics
    packet_loss_count = 0

    try:
        # Sending N ping packets to server
        for sequence_number in range(1, N + 1):
            # Creating ping message with sequence no. & timestamp
            timestamp = time.time()
            message = f'ping {sequence_number} {timestamp}'

            try:
                # Send msg to server
                client_socket_tcp.sendall(message.encode())  # Use sendall for TCP to ensure all data is sent
                
                # Time message was sent, will be recorded
                start_time_tcp = time.time()

                # Wait for server's response
                response = client_socket_tcp.recv(1024)
                
                # Record the time the response was received in end_time_tcp
                end_time_tcp = time.time()
                
                # Calculating the Round trip time
                r_t_t = (end_time_tcp - start_time_tcp) * 1000  # Convert to milliseconds
                
                # Print the server's response and RTT
                print(f'Received Packet #{sequence_number} Which contain: {response.decode()} Having RTT {r_t_t:.2f} ms')
                # Store the RTT for statistics
                rtt_times.append(r_t_t)
            
            except socket.timeout:
                # If the client times out waiting for a response, count it as a packet loss
                print(f'Request timed out for packet #{sequence_number}')
                packet_loss_count += 1

    except (OSError, KeyboardInterrupt) as e:
        # Handle general errors or interruption (e.g., client is killed)
        print(f"Client encountered an error: {e}. Exiting...")
    
    finally:
        # Ensure that statistics are printed and the socket is closed even on an error
        if rtt_times:
            minimum_rtt = min(rtt_times)
            maximum_rtt = max(rtt_times)
            average_rtt = sum(rtt_times) / len(rtt_times)
        else:
            minimum_rtt = maximum_rtt = average_rtt = 0.0  # No successful pings
        
        # Packet loss rate
        packet_loss_rate_tcp = (packet_loss_count / N) * 100
        
        print("\nPing Statistics:")
        print(f'Min RTT: {minimum_rtt:.2f} ms')
        print(f'Max RTT: {maximum_rtt:.2f} ms')
        print(f'Avg RTT: {average_rtt:.2f} ms')
        print(f'Packet Loss: {packet_loss_rate_tcp:.2f}%')

        # Close the socket
        client_socket_tcp.close()

# Main program to execute the client
if __name__ == "__main__":
    # Input: Number of pings to send
    N = int(input("Enter the number of pings to send- "))
    tcp_pinger_client(N)

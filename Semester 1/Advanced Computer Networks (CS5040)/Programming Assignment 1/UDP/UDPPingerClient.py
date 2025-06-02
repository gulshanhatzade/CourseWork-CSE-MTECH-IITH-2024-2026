import time
import socket

server_address = ('172.21.132.110', 11000)  

def udp_pinger_client(N):
    # Createing a UDP socket
    client_socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
     
    client_socket_udp.settimeout(1.0) # 1 second timeout on the socket

    # RTT statistics
    rtt_times = []
    packet_loss_count = 0

    try:
        # Sending N pings to the server
        for sequence_no in range(1, N + 1):
            # Ping message with sequence number and the timestamp
            timestamp = time.time()
            message = f'ping {sequence_no} {timestamp}'
            
            try:
                # Send encoded message to the server
                client_socket_udp.sendto(message.encode(), server_address)
                
                # Time when the message was sent is recorded
                start_time_udp = time.time()

                # Waiting for server's response
                response, _ = client_socket_udp.recvfrom(1024)
                
                # Recording time when response was received
                end_time_udp = time.time()
                
                # Round trip time calculation
                r_t_t = (end_time_udp- start_time_udp) * 1000  # Converting to milli-seconds
                
                # Print server's response and round trip time
                print(f'Received Packet #{sequence_no} Which contain: {response.decode()} Having RTT {r_t_t:.2f} ms')
                
                # Store the RTT for printing statistics
                rtt_times.append(r_t_t)
            
            except socket.timeout:
                # If the client times out waiting for a response for 1 sec, count it as a packet loss
                print(f'Request timed out for packet #{sequence_no}')
                packet_loss_count += 1
                # The client_socket.settimeout(1.0) line ensures that the waits for only 1 second for a response. If no response is received within 
                # that time, the code raises a socket.timeout exception, which is caught in the except block where we print that the request timed out for that specific packet. 


    except (OSError, KeyboardInterrupt) as e:
        # Handle general errors or interruption (e.g., client is killed)
        print(f"Client encountered an error: {e}. Exiting...")
        #  ensures that any general errors or interruptions (e.g., a manual shutdown of the client) are handled gracefully. The program won’t crash unexpectedly, and the socket will be closed properly.
    finally:
        # Ensure that statistics are printed and the socket is closed even on an error
        if rtt_times:
            minimum_rtt = min(rtt_times)
            maximum_rtt = max(rtt_times)
            average_rtt = sum(rtt_times) / len(rtt_times)
        else:
            minimum_rtt = maximum_rtt = average_rtt = 0.0  # No successful pings
        
        # Packet loss rate
        packet_loss_rate = (packet_loss_count / N) * 100
        
        print("\n UDP Ping Statistics:")
        print(f'Min RTT: {minimum_rtt:.2f} mili-second')
        print(f'Max RTT: {maximum_rtt:.2f} mili-second')
        print(f'Avg RTT: {average_rtt:.2f} mili-second')
        print(f'Rate of packet Loss is {packet_loss_rate:.2f}%')

        # Closing the socket
        client_socket_udp.close()

# Main program to execute the client
if __name__ == "__main__":
  #   Prompt the user to enter the number of ping requests to send
    #   and validate the input:
    #   - Ensure the input is an integer.
    #   - Ensure the input is greater than 0.
    #   - Continue prompting until a valid input is provided
    while True:
    # Get the number of pings to send from the user
        try:#
            N = int(input("Enter no of ping requests to the server "))
            if N <= 0:
                print("Invalid input. Please enter a number greater than 0.")
                continue  # Prompt for input again
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
            continue  # Prompt for input again
        break
    udp_pinger_client(N)
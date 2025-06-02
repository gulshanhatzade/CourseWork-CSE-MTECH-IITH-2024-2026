import socket
import threading # to handle multiple clients 

# Defining the server's  IP address and port
server_ip_address = ('192.168.0.222', 11003)  
def handle_client(connectionSocket): # Function designed to run for each client having separate thread
    while True:
        message = connectionSocket.recv(1024)  # Receiving the client packet
        if not message:
            break
        
        message = message.upper()  # Converting clients message to Uppercase
        
        connectionSocket.send(message) #Sending back message in Uppercase to the Client

    # Close the connection socket after handling the client
    connectionSocket.close()
    # After the loop exits (when the client disconnects), the function closes the connectionSocket. 
    # This is important for freeing up resources and properly terminating the connection.

def tcp_pinger_server():
    """Function to set up and run the TCP pinger server"""
    # Create a TCP socket (SOCK_STREAM for TCP)
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind the socket to the IP address and a port number (greater than 10000)
    serverSocket.bind(server_ip_address)
    
    # Listen for incoming connections and sets a maximum queue length of 5 for incoming connections.
    serverSocket.listen(5)
    # 5 specifies the backlog parameter for the TCP socket. This backlog parameter determines the maximum number of queued connections 
    # help maintain performance and stability
    print(f'The server is ready to receive on port {server_ip_address[1]}')

    while True:
        # Wait for a client to connect
        connectionSocket, addr = serverSocket.accept()
        print(f'Connection from {addr} has been established!')
        
        # Create a new thread to handle the client. continues to listen for other connections.
        # The server accepts a connection, starts a new thread to handle that client, and continues to listen for other connections
        client_thread = threading.Thread(target=handle_client, args=(connectionSocket,))
        # threading.Thread creates a new thread.The target parameter specifies the function (handle_client) that will be run in the new thread.
        client_thread.start()
        # starts the execution of the handle_client function in a new thread.

# Main program to execute the server
if __name__ == "__main__":
    tcp_pinger_server()

import random
from socket import *

# Defining the TCP socket, used - SOCK_STREAM for TCP
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

# Binding socket to the IP address and a port no. 11000
serverSocket.bind(('192.168.0.222', 11003))

# Listen for incoming connections
serverSocket.listen(5)
print('Server is ready to receive on port 11000')

# Infinite loop to keep the server running and handling clients
while True:
    # Wait for a client to connect
    connectionSocket, addr = serverSocket.accept() #Connection-Oriented: TCP requires a connection to be established before data transfer,
    print(f'Connection from IP address {addr} has been established.')

    while True:
        # Generate a random number between 1 to 10 (both inclusive)
        rand = random.randint(1, 10)
        
        # Receive the client packet
        message = connectionSocket.recv(1024)
        if not message:
            break
        
        # Capitalize the message from the client
        message = message.upper()
        
        # Simulate packet loss: if rand is greater than 8, do not respond
        if rand > 8:
            continue
        
        # Send back the capitalized message to the client
        connectionSocket.send(message)

    # Close the connection socket after handling the client ensuring no lingering sockets.
    connectionSocket.close()

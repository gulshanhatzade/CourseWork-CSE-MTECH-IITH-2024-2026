from socket import *

# Creating a UDP socket
# Notice the use of SOCK_DGRAM for UDP packets
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

# Assign IP address and port number to socket
serverSocket.bind(('172.21.132.168', 11000))

print("Server is ready to connect and receive...")

while True:
    # Receive the client packet along with the address it is coming from
    message, address = serverSocket.recvfrom(1024)

    # Capitalize the message from the client
    message = message.upper()

    # Send the response back to the client
    serverSocket.sendto(message, address)

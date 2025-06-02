# UDPPingerServer.py
# We will need the following module to generate randomized lost packets
import random
from socket import *
# Creating the  UDP socket
# Notice the use of SOCK_DGRAM for UDP packets
serverSocket = socket(AF_INET, SOCK_DGRAM)
# Assigning the IP address and port no. to socket
serverSocket.bind(('192.168.0.222', 11000))

while True:
# Generate the random number between 1 to 10 (both inclusive)
    rand = random.randint(1, 10)
# Receive the client packet along with the address it is coming from
    message, address = serverSocket.recvfrom(1024)
# Capitalize the message from the client
    message = message.upper()
# If rand is greater than 8, we consider the packet lost and do not respond to the client
    if rand > 8:
       continue
    serverSocket.sendto(message, address) # Otherwise, the server response


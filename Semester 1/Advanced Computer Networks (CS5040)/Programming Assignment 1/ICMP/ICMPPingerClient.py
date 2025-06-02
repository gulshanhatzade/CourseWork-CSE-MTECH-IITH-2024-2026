from socket import * # creating raw sacket to handle the tasks
import os            # To interact with operating system
import sys           # to access sustem specific parameters
import struct        # to handle icmp headers (unpack and pack)
import time          # for RTTs
import select        # to handle sockets
import binascii      # Helps in handling binary data

ICMP_ECHO_REQUEST = 8  # Standard ICMP message used for ECHO Request 

def checksum(string):   # To Calculate Checksum of ICMP Packet
    csum = 0
    countTo = (len(string) // 2) * 2
    count = 0

    while count < countTo:
        thisVal = string[count+1] * 256 + string[count]
        csum = csum + thisVal
        csum = csum & 0xffffffff
        count = count + 2

    if countTo < len(string):
        csum = csum + string[len(string) - 1]
        csum = csum & 0xffffffff

    csum = (csum >> 16) + (csum & 0xffff)
    csum = csum + (csum >> 16)
    answer = ~csum
    answer = answer & 0xffff
    answer = answer >> 8 | (answer << 8 & 0xff00)
    return answer

def receiveOnePing(mySocket, ID, timeout, destAddr):    # Listning for ICMP Echo Replies
    timeLeft = timeout

    while True:
        startedSelect = time.time()    # time from started waiting for the reply
        whatReady = select.select([mySocket], [], [], timeLeft) # select.select monitors socket for incoming data till timeleft seconds
        howLongInSelect = (time.time() - startedSelect)   # for calculating how long select call took (waiting for reply)

        if whatReady[0] == []:  # Timeout as no reply reecived within time limit (timeLeft)
            return "Request timed out."

        timeReceived = time.time()  # storing the time when packet was received from server
        recPacket, addr = mySocket.recvfrom(1024)  # Receiving packets 

        # Fetch the ICMP header(which is from byte 20 to 28 of packet) from the IP packet
        icmpHeader = recPacket[20:28]
        type, code, checksum, packetID, sequence = struct.unpack("bbHHh", icmpHeader) #bbHHh is the format for fields
        
        if packetID == ID and type == 0:   # Checking if the packet ID matches the one we sent and if it Echo Reply(type == 0) 
              # Echo reply
               rtt = (timeReceived - struct.unpack("d", recPacket[28:])[0]) * 1000   # Calculating RTT 
               return f"RTT: {rtt:.2f} ms" # RTT of the Packet

        else:
                 # Handle different ICMP error types
            if type == 3:  # Destination Unreachable
                if code == 0:
                    return "Destination Network Unreachable"
                elif code == 1:
                    return "Destination Host Unreachable"
                elif code == 2:
                    return "Destination Protocol Unreachable"
                elif code == 3:
                    return "Destinationnnnn Port Unreachable"
                else:
                    return "Destination Unreachable (Other)"
            elif type == 11:  # Time Exceeded
                return "Time to live exceeded"
            else:
                return f"ICMP error: Type {type}, Code {code}"
   
        timeLeft = timeLeft - howLongInSelect # calculating time for select fucntion to wait if there any within limit
        if timeLeft <= 0:
            return "Request timed out."

def sendOnePing(mySocket, destAddr, ID):
    # Header is type (8), code (8), checksum (16), id (16), sequence (16)
    myChecksum = 0

    # Make a dummy header with a 0 checksum
    # struct -- Interpret strings as packed binary data
    header = struct.pack("bbHHh", ICMP_ECHO_REQUEST, 0, myChecksum, ID, 1)
    data = struct.pack("d", time.time())

    # Calculate the checksum on the data and the dummy header.
    myChecksum = checksum(header + data)

    # Get the right checksum, and put it in the header
    if sys.platform == 'darwin':
        # Convert 16-bit integers from host to network byte order
        myChecksum = htons(myChecksum) & 0xffff
    else:
        myChecksum = htons(myChecksum)

    header = struct.pack("bbHHh", ICMP_ECHO_REQUEST, 0, myChecksum, ID, 1)
    packet = header + data

    mySocket.sendto(packet, (destAddr, 1))  # AF_INET address must be tuple, not str

def doOnePing(destAddr, timeout):
    icmp = getprotobyname("icmp")
    # SOCK_RAW is a powerful socket type. For more details:
    # http://sockraw.org/papers/sock_raw
    mySocket = socket(AF_INET, SOCK_RAW, icmp)

    myID = os.getpid() & 0xFFFF  # Return the current process ID
    sendOnePing(mySocket, destAddr, myID)
    delay = receiveOnePing(mySocket, myID, timeout, destAddr)

    mySocket.close()
    return delay

def ping(host, timeout=1, count=6):
    # timeout=1 means: If one second goes by without a reply from the server,
    # the client assumes that either the client's ping or the server's pong is lost
    dest = gethostbyname(host)  # converts host name to IP
    print("Pinging " + dest + " using Python:")  # printing the IP 
    print("")

    # Statistics variables
    rtts = []   # creating a list to store RTTs
    packets_sent = 0  # Initialization of packet sent
    packets_received = 0  # Initialization of packet received

    # TO Send multiple pings to a server separated by approximately one second
    for i in range(count):
        packets_sent += 1   # incrementing by 1
        delay = doOnePing(dest, timeout)  # Sending Ping Request
        print(delay)
        
        if "Request timed out" not in delay and "Unreachable" not in delay: # checking for response valid or error
            packets_received += 1
            rtt_value = float(delay.split()[1])  # Extracting RTT value
            rtts.append(rtt_value) # adding RTT value to list
        
        time.sleep(1)  # one second wait before sending next ping

    # Calculating RTTs
    if rtts:
        minimumRTT = min(rtts)
        maximumRTT = max(rtts)
        avgerageRTT = sum(rtts) / len(rtts)
    else:
        minimumRTT = 0
        maximumRTT = 0
        avgerageRTT = 0

    packet_loss = ((packets_sent - packets_received) / packets_sent) * 100  # Calculating Packate loss 

    # Print final statistics of RTTs and Packet loss
    print("\n Statistics about RTTs and Packet loss about Pings ")
    print(f"Minimum RTT = {minimumRTT:.2f} ms")
    print(f"Maximum RTT = {maximumRTT:.2f} ms")
    print(f"Average RTT = {avgerageRTT:.2f} ms")
    print(f"{packets_sent} packets transmitted, {packets_received} packets received, {packet_loss:.1f}% packet loss")

# Example usage:
ping("google.com")

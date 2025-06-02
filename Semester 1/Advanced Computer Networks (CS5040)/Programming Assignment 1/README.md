# Programming Assignment 1

This repository contains Part 1 (UDP Pinger client,server and modified server), Part 2 (TCP Pinger client, server, modified server), Part 3 (ICMP client). Each part involves the implementation of a file in Python.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Part 1- UDP Pinger Client, Server, Modified Server](#part-1-udp-pinger-client)
4. [Part 2- TCP Pinger Client, Server, Modified Server](#part-2-additional-question)
5. [Part 3- ICMP Client](#part-3-additional-question)
6. [Example Output](#example-output)
7.  [Error Handling](#error)


## Overview

This assignment focuses to understand the basic networking principles by implementing various network applications UDP, TCP and ICMP python. 

The first part of the assignment is build to implement a **UDP Pinger Client**, in which we have created UDPPingerClient.py which takes the input for how many pings user wants to send to the server and sends the pings and prints the min , max and avg RTT time in ms with packet loss rate. 

The second part of assignment is implementing the  **TCP Pinger Client** here we implemented the same as in case of UDP with changes required for TCP connection and we also created TCP modified server which does not produces packet loss so we introduced packet loss by tc method.

The last part of assinment is implementing **ICMPPingerClient.py** file, here we printed min, max, avg RTT for the successful pings. We blocked the device and send pings to that device and printed error messages like "Destination Port Unreachable" or "Destination Host Unreachable".

## Requirements

- Python 3.x installed on linux system
- Working knowledge of networking concepts like UDP, TCP, ICMP and sockets.

## Part 1: UDP Pinger 

### UDP Pinger Client

In this part simple UDP pinger client is implemented. Client sends `N` ping requests to the UDP serverand measures the min, avg, max Round Trip Times and also calculates packet loss rate.


### UDP Pinger Client Code
The code for this part is in the `UDPPingerClient.py` file.


### UDP Pinger Modified server

This is the modified server which does not introduces the packet loss so we used tc method for introducing the packet loss of 20%. The following command is used for introducing the packet loss.

```bash
     sudo tc qdisc add dev wlp0s20f3 root netem loss 20%
   ```
For checking the commands which are executed are successfull or run the command.

```bash
     sudo tc qdisc show dev wlp0s20f3
   ```
After executing the client, the 20% packet loss will be deleted by the following command.

```bash
      sudo tc qdisc del dev wlp0s20f3 root
   ```

### UDP Pinger Modified server Code
The code for this part is in the `UDPPingerModifiedServe.py` file.

### Features
- Sends N (input taken from user) pings to the server.
- Prints RTT in milli-second for each ping send.
- Reports lost packets with their ping number.
- Shows rate of packet loss.

### How It Works
- UDP socket is created and `N` pings are sent to a server at IP address `172.21.132.168` on the port `11000`.
- Each ping contains sequence no. and the timestamp.
- If their is response from server side, the RTT is  recorded. If server does not respond within time limit which is 1 second, it will be marked as time out.
- At the end Round Trip Time statistics and rate of packet loss will be printed on client's terminal.


## Example Output

Here’s an example of the output from the UDP Pinger Client:

```plaintext
Enter no of ping requests to the server 26

Request timed out for packet #1

Received Packet #2 Which contain:

Received Packet #3 Which contain: Request timed out for packet #4 PING 3 1725902830.0489087 Having RTT 2.16 ms

Received Packet #5 Which contain: PING 5 1725902831.052578 Having RTT 121.14 ms

Received Packet #6 Which contain: PING 6 1725902831.1739087 Having RTT 2.70 ms

Received Packet #7 Which contain: PING 7 1725902831.176783 Having RTT 1.69 ms

Received Packet #8 Which contain: PING 8 1725902831.178618 Having RTT 2.49 ms

Received Packet #9 Which contain: PING 9 1725902831.18126 Having RTT 1.60 ms Received Packet #10 Which contain: PING 10 1725902831.1829677 Having RTT 1.40 ms

Received Packet #11 Which contain: PING 11 1725902831.1844604 Having RTT 1.84 ms Request timed out for packet #12

PING 2 1725902829.9374495 Having RTT 111.30 ms

Received Packet #13 Which contain: PING 13 1725902832.1873472 Having RTT 214.14 ms Received Packet #14 Which contain: PING 14 1725902832.401621 Having RTT 206.45 ms

Received Packet #15 Which contain: PING 15 1725902832.6882304 Having RTT 191.24 ms

Received Packet #16 Which contain: PING 16 1725902832.79966 Having RTT 2.10 ms.

Request timed out for packet #17 Request timed out for packet #18

Received Packet #19 Which contain: PING 19 1725902834.804823 Having RTT 157.21 ms

Request timed out for packet #20

Received Packet #21 Which contain: PING 21 1725902835.9633694 Having RTT 27.26 ms

Request timed out for packet #22

Request timed out for packet #23

Request timed out for packet #24

Received Packet #25 Which contain: PING 25 1725902838.9933732 Having RTT 264.37 ms

Received Packet #26 Which contain: PING 26 1725902839.2579079 Having RTT 1.71 ms

UDP Ping Statistics:

Min RTT: 1.40 mili-second

Max RTT: 264.37 mili-second

Avg RTT: 77.11 mili-second

Rate of packet Loss is 34.62%
```














## Part 2: TCP Pinger

### TCP Client
In this part simple TCP pinger client code is implemented. Client sends N ping requests to the TCP server and measures the minimum, maximum and the average Round Trip Times and also calculates packet loss rate. Client code introduces the loss of 20% by random integer method.

### TCP Pinger Client Code
The code for this part is in the `TCPPingerClient.py` file.


### TCP Pinger Modified server

This is the modified server which does not introduces the packet loss so we used tc method for introducing the packet loss of 20%. The following command is used for introducing the packet loss. This modified sever can handle concurrent client at a time, multiple clients can able to establish the connection and send pings to multi threaded server. Even if one of the client terminates in between still it can handle other clients.

```bash
     sudo tc qdisc add dev wlp0s20f3 root netem loss 20%
   ```
For checking the commands which are executed are succesfull or run the command.

```bash
     sudo tc qdisc show dev wlp0s20f3
   ```
After executing the client, the 20% packet loss will be deleted by the following command.

```bash
      sudo tc qdisc del dev wlp0s20f3 root
   ```

### TCP Pinger Modified server Code
The code for this part is in the `TCPPingerModifiedServe.py` file.

### Features
- Sends N (input taken from user) pings to the server by TCP connection
- Prints Round Trip Time in milli-second for each ping send.
- Reports lost packets with their ping number.
- Shows rate of packet loss.

### How It Works
- TCP socket is created and `N` pings are sent to a server at IP address `192.168.0.222` on the port `11000`.
- Each ping contains sequence no. and the timestamp.
- If their is response from server side, the RTT is  recorded. If server does not respond within time limit which is 1 second, it will be marked as time out.
- At the end RTT statistics and rate of packet loss will be printed on client's terminal.
- TCP modified server be handle concurrent clients.

## Example Output

Here’s an example of the output from the TCP Pinger Client:

```plaintext
Enter the number of pings to send: 20

Received Packet #1 Which contain: PING 1 1725904816.513799 Having RTT 203.16 ms

Received Packet #2 Which contain: PING 2 1725904816.7172258 Having RTT 2.37 ms

Received Packet #3 Which contain: PING 3 1725904816,7198036 Having RTT 586.21 ms

Received Packet #4 Which contain: PING 4 1725994817.3061748 Having RTT 1.52 ms

Received Packet #5 Which contain: PING 5 1725904817.307873 Having RTT 1.63 ms

Received Packet #6 Which contain: PING 6 1725904817.3096297 Having RTT 1.69 ms

Received Packet #7 Which contain: PING 7 1725904817.3114424 Having RTT 2.71 ms

Request timed out for packet #8 Request timed out for packet #9

Received Packet #10 Which contain: PING 8 1725904817.3142834 Having RTT 37.45 ms 1725904818.3156116PING 10 1725904819.3169534 Having RTT 2.86 ms.

Received Packet #11 Which contain: PING 9

Received Packet #12 Which contain: PING 11 1725904819.3546085 Having RTT 1.16 ms

Received Request timed out for packet #14

Packet #13 Which contain: PING 12 1725904819.3575585 Having RTT 1.19 ms

Received Packet #15 Which contain: PING 13 1725904819.3588033PING 14 1725964819.360075 Having RTT 656.22 ms

Received Packet #16 Which contain: PING 15 1725904820.3617315PING 16 1725904821.018895 Having RTT 712.60 ms

Received Packet #17 Which contain: PING 17 1725904821.7308753 Having RTT 2.08 ms Received Packet #18 Which contain: PING 18 1725904821.733108 Having RTT 2.16 ms

Received Packet #19 Which contain: PING 19 1725904821.735422 Having RTT 1.56 ms

Received Packet #20 Which contain: PING 20 1725904821.7371254 Having RTT 1.80 ms

Ping Statistics:

Min RTT: 1.16 ms

Max RTT: 712.60 ms

Avg RTT: 130.49 ms Packet Loss: 15.00%
 ```










## Part 3: ICMP Client

### ICMP Client
In this part simple ICMP pinger client is implemented. For this we had the requirement of if packets are matching, we had to fetch the information from packet header such as type,checksum,etc and we had to print the RTTs of each ping request along with conclusion of min,max and avg RTT amongs them, also packet loss rate. 

Run this command on server device to block the client IP address-

```bash
     sudo iptables -A INPUT -s 172.21.132.110 -p icmp -j REJECT icmp-host-unreachable
   ```
After performing the task delete the reject operation which done by above command by-

```bash
    sudo iptables -F
   ```


### Features
- This calculates min, max, average RTT and packet loss rate.
- Prints the round trip time.
- Gives error messages like "Destination Host Unreachable" or "Destination Host Unreachable".

### How It Works
Execute this commands in terminal
```bash
    sudo python3 ICMPPingerCLient.py
   ```



### TCP Pinger Client Code
The code for this part is in the `ICMPPingerClient.py` file.


## Example Output

Here’s an example of the output from the TCP Pinger Client:

```plaintext
ankitonlinux@fedora:-/python$ sudo python ICMPPingerClient.py Pinging 192.168.0.237 using Python:

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Destination Host Unreachable

Ping statistics

20 packets transmitted, 0 packets received, 100.0% packet loss rtt min/avg/max = 0.00/0.00/0.00 ms
```



```plaintext
ankitonlinux@fedora:-/python$ sudo python ICMPPingerClient.py

Pinging 192.168.0.237 using Python:

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

Destinationnnnn Port Unreachable

20 packets transmitted, 0 packets received, 100.0% packet loss rtt min/avg/max = 0.00/0.00/0.00 ms
```


```plaintext
ankitonlinux@fedora:-/python$ sudo python ICMPPingerClient.py Pinging 142.250.193.142 using Python:

RTT: 93.43 ms

RTT: 31.59 ms

RTT: 83.75 ms

RTT: 69.47 ms

RTT: 28.05 ms

RTT: 118.87 ms

RTT: 125.61 ms

RTT: 118.45 ms

RTT: 132.41 ms

RTT: 126.37 ms

RTT: 34.31 ms

RTT: 114.50 ms

---Ping statistics

Minimum RTT 28.05 ms 132.41 ms

Maximum RTT

Average RTT 89.73 ms

12 packets transmitted, 12 packets received, 0.0% packet loss
```








## Error Handling
Exceptions: Code is designe flr handling errors such as network issues, socket errors and manual interruption like Ctrl+C.

Timeouts: Client detects when a ping request times out after 1 second and  then reports it as packet is lost.









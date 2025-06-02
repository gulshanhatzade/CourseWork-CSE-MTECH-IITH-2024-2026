Programming Assignment 1 CS6903: Network Security

# Task 2 - Lets Chit-Chat !!! Chat Between Two Users - ChatBuddy: A Simple Peer-to-Peer Chat System

## Project Description
ChatBuddy is a simple peer-to-peer (P2P) chat system that allows users to connect and communicate with each other over a network. The system includes a central server (CentralHub) that maintains a list of online users and a client application (MessageBuddy) that allows users to start and receive chat sessions.

## Team Members
1. CS24MTECH14003 - Anurag Sarva
2. CS24MTECH14006 - Gulshan Hatzade

---

## Project Components
The project consists of two main components:

### 1. Server (CentralHub)
- Manages user registrations and maintains a list of connected users.
- Handles client requests to retrieve a list of online users.
- Runs on a fixed port to accept connections from multiple clients.

### 2. Client (MessageBuddy)
- Connects to the central server to register and fetch online users.
- Allows users to initiate or receive peer-to-peer chat sessions.
- Supports basic message exchange between users.

---

## Installation & Execution Steps
### **Prerequisites**
- Python 3.x installed on your system.
- Basic understanding of socket programming in Python.

### **Step 1: Start the Server**
1. Open a terminal and navigate to the project directory.
2. Run the following command to start the server:
            python server.py
3. The server will start listening for incoming client connections on port **8080**.


### **Step 2: Start the Client**
1. Open a separate terminal and navigate to the project directory.
2. Run the client program using:
             python client.py
3. When prompted, enter your desired username.
4. You will see a menu with the following options:
   - **1. See who's online** → Lists available users.
   - **2. Start a chat with someone** → Enter the IP address and port of another user to start a conversation.
   - **3. Wait for someone to chat** → Listens for incoming chat requests.
   - **4. Exit program** → Closes the application.
   - **5. Start new client by doing same


### **Step 3: Establish a Chat Connection**
1. One user selects option **2** to initiate a chat and enters the friend's IP and port.
2. The other user selects option **3** to wait for an incoming connection.
3. Once connected, users can exchange messages.
4. To exit the chat, type **EOM** and press Enter.

### **Step 4: Stop the Server**
- To gracefully shut down the server, press **Ctrl + C** in the server terminal.

---

## Example Usage
### **User 1 (Client A) starts the client**

 python client.py
Successfully connected to main server!
What should we call you?: Alice
Name registered successfully.
What would you like to do?
1. See who's online
2. Start a chat with someone
3. Wait for someone to chat
4. Exit program
Choose your option (1-4): 2
Enter their address: 127.0.0.1
Enter their port: 9090
Trying to reach your friend at 127.0.0.1:9090...
Connection successful! Chat away!
When you're done, just type 'EOM'
[Your turn]: Hello!
```

### **User 2 (Client B) waits for incoming chat**

 python client.py
Successfully connected to main server!
What should we call you?: Bob
Name registered successfully.
What would you like to do?
1. See who's online
2. Start a chat with someone
3. Wait for someone to chat
4. Exit program
Choose your option (1-4): 3
Waiting for someone to start a chat...
Ready for chat on port 9090...
Someone joined! Start chatting!
Remember to type 'EOM' when you're finished
[Friend says]: Hello!
[Your turn]: Hi Alice! How are you?
```
-----------------------------------------------------------------
Example:-
-----------------------------------------------------------------
......................................
server -
......................................
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 2$ python3 server.py 
Server is now running!
Accepting connections on port 8080...
Can handle up to 100 simultaneous users

New connection from 192.168.198.111:32780
New user 'Anurag' has joined.
New connection from 192.168.198.99:59012
New user 'Gulshan' has joined.
User 'Anurag' has disconnected.
User 'Gulshan' has disconnected.
^C
Server shutting down...

......................................
client 1
......................................
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 2$ export P=23
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 2$ export Q=5
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 2$ python3 client.py 
Getting everything ready...

Successfully connected to main server!

What should we call you?: 
Anurag
Name registered successfully.

What would you like to do?
1. See who's online
2. Start a chat with someone
3. Wait for someone to chat
4. Exit program
Choose your option (1-4): 
3

Waiting for someone to start a chat...
Ready for chat on port 9090...
Someone joined! Start chatting!
Remember to type 'EOM' when you're finished

[Send]: 

[Received]: Hii Anurag
[Send]: 

[Send]: 
Hii Gulsahn
[Send]: 

[Received]: How are you bro ?
[Send]: 
i am fine 
[Send]: 
and what about you?
[Send]: 
i have to go 
[Send]: 
byee
[Send]: 
EOM

What would you like to do?
1. See who's online
2. Start a chat with someone
3. Wait for someone to chat
4. Exit program
Choose your option (1-4): 
4
Thanks for using ChatBuddy!

......................................
client 2
......................................
(base) gulshanhatzade@Gulshans-MacBook-Air Task 2 % export P=23      
(base) gulshanhatzade@Gulshans-MacBook-Air Task 2 % export Q=5       
(base) gulshanhatzade@Gulshans-MacBook-Air Task 2 % python3 client.py
Getting everything ready...

Successfully connected to main server!

What should we call you?: 
Gulshan
Name registered successfully.

What would you like to do?
1. See who's online
2. Start a chat with someone
3. Wait for someone to chat
4. Exit program
Choose your option (1-4): 
1

Checking who's online...

=== Currently Online Users ===
Username: Anurag
Address: 192.168.198.111
-------------------
Username: Gulshan
Address: 192.168.198.99
-------------------


What would you like to do?
1. See who's online
2. Start a chat with someone
3. Wait for someone to chat
4. Exit program
Choose your option (1-4): 
2
Use this port to coonet with the peer 
9090
Enter their address: 
192.168.198.111
Enter their port: 
9090

Trying to reach your friend at 192.168.198.111:9090...
Connection successful! Chat away!
When you're done, just type 'EOM'

[Send]: 
Hii Anurag
[Send]: 

[Received]: Hii Gulsahn
[Send]: 
How are you bro ?
[Send]: 

[Received]: i am fine 
[Send]: 

[Received]: and what about you?
[Send]: 

[Received]: i have to go 
[Send]: 

[Received]: byee
[Send]: 

[Received]: EOM
[Send]: 
E
Looks like your friend stepped away.
EOM
[Send]: 
EOM
Something went wrong with the chat: [Errno 32] Broken pipe

What would you like to do?
1. See who's online
2. Start a chat with someone
3. Wait for someone to chat
4. Exit program
Choose your option (1-4): 
4
Thanks for using ChatBuddy!
(base) gulshanhatzade@Gulshans-MacBook-Air Task 2 %

------------------------------------------------------------
## Error Handling
------------------------------------------------------------
- If a user enters an invalid IP or port, an error message is displayed.
- If the server is down, clients will not be able to register or fetch online users.
- If a connection is lost, an appropriate message is shown.

---

## Conclusion
ChatBuddy provides a simple and interactive way to communicate using a centralized server for user discovery and a P2P architecture for message exchange. It demonstrates the practical use of sockets and multithreading in Python.

---

ANTI-PLAGIARISM STATEMENT -

We certify that this assignment/report is our own work, based on our personal study and/or research
and that we have acknowledged all material and sources used in its preparation, whether books, articles,
packages, datasets, reports, lecture notes, or any other document, electronic or personal communication.
We also certify that this assignment/report has not previously been submitted for assessment/project in any
other course lab, except where specific permission has been granted from all course instructors involved, or
at any other time in this course, and that we have not copied in part or whole or otherwise plagiarized the
work of other students and/or persons. We pledge to uphold the principles of honesty and responsibility at
CSE@IITH. In addition, We understand my responsibility to report honor violations by other students if
we become aware of it.

Names:  Anurag Sarva, Gulshan Hatzade
Date:   04/02/2025
Signature: Anurag Sarva, Gulshan Hatzade



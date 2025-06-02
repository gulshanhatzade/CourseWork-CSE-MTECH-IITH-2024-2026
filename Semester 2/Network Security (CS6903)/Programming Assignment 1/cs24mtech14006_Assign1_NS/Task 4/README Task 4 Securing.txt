Task 4: Securing the Chat Communication Using AES
------------------------------------------------------------
Project Overview
------------------------------------------------------------
Task 4 is focuses on securing the chat communication by integrating AES encryption into our P2P chat application. In this task, messages exchanged between users are encrypted using AES in CBC mode with a key derived from a DiffieHellman key exchange. The implementation builds on previous tasks and consists of the following components -

1. server.py
   - Implementing a central chat server.
   - Managing client connections and maintains a dictionary mapping user IDs to their (IP, port) details.
   - Provides the current list of connected users on the basis of requests.

2. dh_utils.py
   - Contains functions for DiffieHellman key exchange.
   - Note: In this Task 4 code, the DiffieHellman related functionality is used by the client module, and the utilities are imported from dh_utils.
   
3. crypto_utils.py
   - Provides methods for encrypting and decrypting messages using AES in CBC mode.
   - Encrypts plaintext messages and outputs a base64 encoded string (combining the IV and ciphertext).
   - Decrypts received encrypted messages back into the plaintext.

4. client.py
   - Implements the secure chat client.
   - Reads DiffieHellman parameters P and Q from environment variables.
   - Connects to the central server to register and retrieve the list of online users.
   - Initiates and accepts the chat sessions with peers.
   - Performs a DiffieHellman key exchange with a peer to derive a shared AES key.
   - Encrypts outgoing messages and decrypts incoming messages, displaying both the ciphertext and the decrypted plaintext.

------------------------------------------------------------
Environment Setup
------------------------------------------------------------
Before running the client, set the DiffieHellman parameters as environment variables. For example, in a Linux terminal:
   export P=9973        # A large prime number
   export Q=5           # A generator (g can be anything from 1 to P)

Ensure that the pycryptodome library is installed for AES functionality. You can install it using:
   pip install pycryptodome

------------------------------------------------------------
Execution Instructions
------------------------------------------------------------
Step 1: Start the Server
   1. Open a terminal and navigate to the directory containing server.py.
   2. Run the server using:
         python3 server.py
   3. The server will bind to port 8080 and start listening for incoming connections.
   4. You should see a message indicating that the server is starting.
   (Note: if you are running client and server on different systems make sure us the IP of the server running system and also make sure connected with the same network.)

Step 2: Start the Client
   1. Open another terminal and navigate to the directory containing client.py.
   2. Ensure that the environment variables P and Q are set as described above.
   	 like - export P=23 # must be prime number.
   	 	export Q=5
   3. Run the client using:
         python3 client.py
   4. When prompted, enter your user name.
   5. The client will connect to the central server, register your details, and retrieve the current list of online users.(Follow same Step 2, for connecting second client.)

Step 3: Secure Chat Session
   1. From the client menu, choose the appropriate option:
         - Option 1: List users – displays the current connected users.
         - Option 2: Connect to a user – prompts for a target user name to initiate a secure chat.
         - Option 3: Exit – (closes the client application.)
   2. When initiating a chat:
         - The client performs a DiffieHellman key exchange with the peer.
         - Public keys are exchanged and a shared secret is computed.
         - An AES key is derived from the shared secret and printed on the console.
   3. During the chat session:
         - Outgoing messages are encrypted using the derived AES key.
         - The encrypted message i.e. ciphertext is displayed with the label “[Cipher]”.
         - Upon receiving a message, the client decrypt it and displays the plaintext with the label “[Plain]”.
         - To terminate the chat session, type “EOM”.

Step 4: Exiting the Client
   - Select the option to exit from the main menu.
   - The client will close the connection to the central server and shut down gracefully.

------------------------------------------------------------
Example Usage
------------------------------------------------------------
Starting the Server:


anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 4$ python3 server.py 
Server starting on 192.168.155.111:8080
Server is running ing...
User Anurag is connected from 192.168.155.111:57025
User Gulshan is connected from 192.168.155.99:60259
User Anurag is disconnected
User Gulshan is disconnected
^C
Server shutting down
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 4$ 




Starting the Client 1:



anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 4$ export P=23
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 4$ export Q=5
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 4$ python3 client.py 
Enter your userID: Anurag

Options:
1. List users
2. Connect to a user
3. Exit
Choice: [Received]: Public Key = 3
[Send]: Public Key = 21
Generated AES Key: 19581e27de7ced00ff1ce50b2047e7a5

Chat starting! Type 'EOM' to end ing chat.

[Cipher]: 3V5sFck5dQoICPRQv8Dk4rrpaxPu3XxDjAdpPi0yeVU=
[Plain]: Hii Anurag
[Send]: 
[Send]: Hi gulsahn
[Cipher]: cxQwT54U1Iit6fqPjOYOdhgmkiKoIxrW9bKZnrFjjUU=
[Send]: 
[Cipher]: aUsYfhaPHR+3+UB5H6dQwjnJgb/01SyKQw5SZCdzf18=
[Plain]: whats up anurag
[Send]: i am fine and waht about you
[Cipher]: 20urWg/8YlApK6DRT8oOFCnJQlilQ2wcllLr52v/isiJ+6BXhF6/Ugxl2WlkdIsX
[Send]: of byeee
[Cipher]: 8nyVNOb/d+uyasm/X5D4rQVZYSXIWgNlmbg3Ds8xtqs=
[Send]: EOM

Chat session ended

Options:
1. List users
2. Connect to a user
3. Exit
Choice: 3
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 4$



Starting the Client 2:




(base) gulshanhatzade@Gulshans-MacBook-Air Task 4 % export P=23      
(base) gulshanhatzade@Gulshans-MacBook-Air Task 4 % export Q=5       
(base) gulshanhatzade@Gulshans-MacBook-Air Task 4 % python3 client.py
Enter your userID: Gulshan

Options:
1. List users
2. Connect to a user
3. Exit
Choice: 1

Current connected users:
- Anurag

Options:
1. List users
2. Connect to a user
3. Exit
Choice: 2
Enter user name to connect with him/her: Anurag
[Send]: Public Key = 3
[Received]: Public Key = 21
Generated AES Key: 19581e27de7ced00ff1ce50b2047e7a5

Chat starting! Type 'EOM' to end ing chat.
[Send]: Hii Anurag
[Cipher]: 3V5sFck5dQoICPRQv8Dk4rrpaxPu3XxDjAdpPi0yeVU=
[Send]: 
[Cipher]: cxQwT54U1Iit6fqPjOYOdhgmkiKoIxrW9bKZnrFjjUU=
[Plain]: Hi gulsahn
[Send]: whats up anurag
[Cipher]: aUsYfhaPHR+3+UB5H6dQwjnJgb/01SyKQw5SZCdzf18=
[Send]: 
[Cipher]: 20urWg/8YlApK6DRT8oOFCnJQlilQ2wcllLr52v/isiJ+6BXhF6/Ugxl2WlkdIsX
[Plain]: i am fine and waht about you
[Send]: 
[Cipher]: 8nyVNOb/d+uyasm/X5D4rQVZYSXIWgNlmbg3Ds8xtqs=
[Plain]: of byeee
[Send]: 
Peer ended ing chat

[Cipher]: PgDOqpC3OXPf2G55uAvcftvzaVOBLuBNsWK0JL9+ws4=

Chat session ended

Options:
1. List users
2. Connect to a user
3. Exit
Choice: 3
(base) gulshanhatzade@Gulshans-MacBook-Air Task 4 %








During a Chat Session:
   [Send]: Public Key = 123456
   [Received]: Public Key = 7891011
   Generated AES Key: a1b2c3d4e5f67890abcdef1234567890
   Chat starting! Type 'EOM' to end ing chat.
   [Send]: Hello, Bob!
   [Cipher]: (encrypted message in base64)
   [Plain]: Hello, Bob!
   (User types “EOM” to end the session.)

------------------------------------------------------------
Contribution of Group Members
------------------------------------------------------------
- Anurag Sarva (CS24MTECH14003):
     • Implemented server.py and contributed to dh_utils.py.
     • Focused on setting up the central server and the DiffieHellman key exchange logic.

- Gulshan Hatzade (CS24MTECH14006):
     • Implemented crypto_utils.py and client.py.
     • Focused on integrating AES encryption/decryption into the chat client and handling secure message transmission.

------------------------------------------------------------
Acknowledgment
------------------------------------------------------------
This project uses the custom implementations for the DiffieHellman key exchange and AES encryption. The AES encryption will leverages the pycryptodome library for cryptographic operations. All functionality has been implemented according to the assignment constraints.

------------------------------------------------------------
Anti-Plagiarism Statement
------------------------------------------------------------
We certify that this assignment/report is our own work, based on our personal study and/or research and that we have acknowledged all material and sources used in its preparation, whether books, articles, packages, datasets, reports, lecture notes, or any other document, electronic or personal communication. We also certify that this assignment/report has not previously been submitted for assessment/project in any other course lab, except where specific permission has been granted from all course instructors involved, or at any other time in this course, and that we have not copied in part or whole or otherwise plagiarized the work of other students and/or persons. We pledge to uphold the principles of honesty and responsibility at CSE@IITH. In addition, We understand my responsibility to report honor violations by other students if we become aware of it.

Names:  Anurag Sarva, Gulshan Hatzade
Date:   04/02/2025
Signature: Anurag Sarva, Gulshan Hatzade


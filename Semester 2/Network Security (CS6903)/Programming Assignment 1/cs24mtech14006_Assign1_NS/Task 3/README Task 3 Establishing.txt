Task 3: Establishing a Shared Secret Using DiffieHellman and AES Encryption
------------------------------------------------------------
Project Overview
------------------------------------------------------------
This task focuses on securely establishing a shared secret between two users using the DiffieHellman key exchange protocol, and then deriving an AES key from the shared secret for secure communication. The implementation includes several modules that work together:

1. server.py
   - Sets up a chat server that handles multiple users.
   - Listens for incoming connections and maintains an active list of users in a dictionary.
   - Sends the current list of online peers to a newly connected user.

2. dh_utils.py
   - Implements functions to generate a random private secret, calculate the public key, and compute the shared secret.
   - Converts the shared secret into a 16-byte AES key using SHA-256 hashing.

3. crypto_utils.py
   - Provides functions to encrypting and decrypting messages using AES in CBC mode.
   - Uses padding and base 64 encoding to manage encryption as well as decryption properly.

4. client.py
   - Implements a secure chat client that connects to the central server.
   - Performs DiffieHellman key exchange with a chat partner to derive a shared AES key.
   - Secures the chat session by encrypting outgoing messages and decrypting incoming messages using the derived key.
------------------------------------------------------------
Environment Setup
------------------------------------------------------------
Before running the client, ensure that the DiffieHellman parameters are set as environment variables:

   P: is large prime number.
   Q: The base (generator) for the DiffieHellman exchange (g is any number from 1 to P).

Example (Linux):
   export P=23
   export Q=5

Make sure your system will have this pycryptodome, if not install it
   pip install pycryptodome
------------------------------------------------------------
Execution Instructions
------------------------------------------------------------
Step 1: Start the Server
   1. Open a terminal and navigate to the directory containing the server code.
   2. Run the following command:
         python3 server.py
   3. The server will start listening on port 8080 and display a message confirming it is running.

Step 2: Start the Client
   1. Open another terminal and navigate to the directory containing the client code.
   2. Ensure the environment variables (P and Q) are set. By writing two command on terminal before client execution-
   	 export P=23
   	 export Q=5(like this)
   3. Run the following command:
         python3 client.py
   4. When prompted, enter your nickname. The client will:
         - Connect to the central server and register your nickname along with the listening port.
         - Retrieve the list of online users.(follow same step, step 2 for connecting second client).

Step 3: Establish a Secure Chat Session
   1. From the client menu, choose the option to see who is online or initiate a chat with a specific user.
   2. When initiating a chat, the client connects to the chosen peer's IP and port.
   3. The Diffie-Hellman key exchange is performed:
         - Each side generates its private secret and calculates its public key.
         - Public keys are exchanged and the shared secret is computed.
         - The shared secret is converted into a 16-byte AES key.
         - The AES key is printed to the console to verify consistency between both parties.
   4. The secure chat session then begins. All messages are encrypted using the AES key before transmission and decrypted upon receipt.
   5. To end the chat session, type "EOM".

Step 4: Exiting the Client
   - From the client menu, select the option to exit. This will close all connections gracefully.

------------------------------------------------------------
Example Usage
------------------------------------------------------------



Starting the Server:



anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 3$ python3 server.py 
Yo! Server is now running on port 8080...
New user joined: Anurag with ip address and port 192.168.155.111:39961
New user joined: Gulshan with ip address and port 192.168.155.99:60184
Seeaa! Anurag left the chat
Seeaa! Gulshan left the chat
^C
(Server Shutting down)..




Starting the Client 1:



anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 3$ export P=23
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 3$ export Q=5
anurag-sarva@anurag-sarva-Inspiron-15-5518:~/NS/NS1/Task 3$ python3 client.py 
What's ur name? Anurag

Select one option which you want to do?
1. See who's here
2. Chat with someone
3. Exit!
Pick one: [Send]: Public Key = 8
[Received]: Public Key = 14
AES Key: 4e07408562bedb8b60ce05c1decfe3ad

Secure chat ready! Type 'EOM' to terminate the chat
[Send]: 
[Received]: HII Anurag
[Send]: 
Hi bro
[Send]: whats upp bro
[Send]: 
[Received]: I am good
[Send]: ok bro byee
[Send]: EOM

Select one option which you want to do?
1. See who's here
2. Chat with someone
3. Exit!
Pick one: 3(similar for the Client 2)





Starting the Client 2-




⚫ (base) gulshanhatzade@Gulshans-MacBook-Air Task 3% export P=23 (base) gulshanhatzade@Gulshans-MacBook-Air Task 3 % export Q=5 (base) gulshanhatzade@Gulshans-MacBook-Air Task 3% python3 client.py What's ur name? Gulshan
Select one option which you want to do?
1. See who's here
2. Chat with someone
3. Exit!
Pick one: 1
Peeps online:
Anurag
Select one option which you want to do?
1. See who's here
2. Chat with someone
3. Exit!
Pick one: 2
Who u wanna chat with? Anurag
[Send]: Public Key = 8
[Received]: Public Key = 14
AES Key: 4e07408562bedb8b60ce05c1decfe3ad
Secure chat ready! Type 'EOM' to terminate the chat
[Send]: HII Anurag
[Send]:
[Received]: Hi bro
[Send]:
[Received]: whats upp bro
[Send]: I am good
[Send]:
[Received]: ok bro byee
[Send]:
Friend is lefting the chat
Select one option which you want to do?
1. See who's here
2. Chat with someone
3. Exit!
EOM
Pick one:
Uh oh, sending failed: [Errno 9] Bad file descriptor EOM
Select one option which you want to do?
1. See who's here
2. Chat with someone
3. Exit!
Pick one: 3




------------------------------------------------------------
Contribution of Group Members
------------------------------------------------------------
- Anurag Sarva (CS24MTECH14003):
     • Implemented server.py and dh_utils.py.
     • Focused on setting up the chat server and the DiffieHellman key exchange mechanism.
     
- Gulshan Hatzade (CS24MTECH14006):
     • Implemented crypto_utils.py and client.py.
     • Focused on integrating AES encryption/decryption and handling client-side secure communication.

------------------------------------------------------------
Acknowledgment
------------------------------------------------------------
This project uses custom implementations for the DiffieHellman key exchange and AES encryption. The AES encryption leverages the pycryptodome library for low-level cryptographic operations, as allowed by the assignment constraints.

------------------------------------------------------------
Anti-Plagiarism Statement
------------------------------------------------------------
We certify that this assignment/report is our own work, based on our personal study and/or research and that we have acknowledged all material and sources used in its preparation, whether books, articles, packages, datasets, reports, lecture notes, or any other document, electronic or personal communication. We also certify that this assignment/report has not previously been submitted for assessment/project in any other course lab, except where specific permission has been granted from all course instructors involved, or at any other time in this course, and that we have not copied in part or whole or otherwise plagiarized the work of other students and/or persons. We pledge to uphold the principles of honesty and responsibility at CSE@IITH. In addition, We understand my responsibility to report honor violations by other students if we become aware of it.

Names:  Anurag Sarva, Gulshan Hatzade
Date:   04/02/2025
Signature: Anurag Sarva, Gulshan Hatzade


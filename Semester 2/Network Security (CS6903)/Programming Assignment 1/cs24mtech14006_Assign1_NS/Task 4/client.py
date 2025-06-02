import socket
import threading
import json
import os
import sys
from dh_utils import DiffieHellmanUtils
from crypto_utils import CryptUtil

# This class is for the secure chat client, dont mind grammar mistakes
class SecChatClient:
    def __init__(self, user_id, srv_host='192.168.155.111', srv_port=8080):
        # Getting p and g from environment variables, its using os.getenv, dont forget
        self.modulus = int(os.getenv('P'))
        self.generator = int(os.getenv('Q'))
        self.user_id = user_id  # user id of client
        self.srv_host = srv_host
        self.srv_port = srv_port
        self.keep_running = False
        self.chat_active = False
        self.chat_mutex = threading.Lock()
        self.aes_secret = None  # will be set after diffie-hellman exchange
        
        # Create a socket to listen for incoming chats
        self.listenSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listenSock.bind(('', 0))
        self.my_listen_port = self.listenSock.getsockname()[1]
        self.listenSock.listen(5)
        
        # Now, connect to the main server
        self.srvSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srvSock.connect((srv_host, srv_port))
        self.srvSock.send(f"{user_id}:{self.my_listen_port}".encode())
        
        # Getting the initial list of users
        self.user_list = json.loads(self.srvSock.recv(1024).decode())
        self.keep_running = True
        
        # Start a thread for accepting incoming chat connection, dont use for other things
        self.accept_thread = threading.Thread(target=self.accept_incoming, daemon=True)
        self.accept_thread.start()

    # This method performs the diffie-hellman exchange with a peer socket
    def do_dh_exchange(self, peerSock, initiating):
        # making a private key using diffie-hellman utils
        priv_key = DiffieHellmanUtils.generate_private_key()
        pub_key = DiffieHellmanUtils.compute_public_key(priv_key, self.generator, self.modulus)
        
        if initiating:
            print(f"[Send]: Public Key = {pub_key}")
            peerSock.send(str(pub_key).encode())
            # waiting for the peer public key
            peer_pub = int(peerSock.recv(1024).decode())
            print(f"[Received]: Public Key = {peer_pub}")
        else:
            # if its not initiating then wait first for the public key from peer
            peer_pub = int(peerSock.recv(1024).decode())
            print(f"[Received]: Public Key = {peer_pub}")
            print(f"[Send]: Public Key = {pub_key}")
            peerSock.send(str(pub_key).encode())
        
        # now compute the shared secret and derive the aes key from it
        shared_sec = DiffieHellmanUtils.compute_shared_secret(priv_key, peer_pub, self.modulus)
        self.aes_secret = DiffieHellmanUtils.derive_aes_key(shared_sec)
        print(f"Generated AES Key: {self.aes_secret}")

    # Method to send encrypted message to the peer
    def transmit_msg(self, peerSock, msg_text):
        try:
            if msg_text.upper() == "EOM":
                peerSock.send("EOM".encode())
                return False
            
            # encrypt the message using our aes key
            encrypted_msg = CryptUtil.encrypt_message(self.aes_secret, msg_text)
            if encrypted_msg:
                peerSock.send(encrypted_msg.encode())
                print(f"[Cipher]: {encrypted_msg}")
            return True
        except Exception as ex:
            print(f"Error on sending msg: {ex}")
            return False

    # This is the main chatting session handler
    def chat_session(self, peerSock, initiating):
        try:
            self.chat_active = True
            # Perform diffie-hellman exchange, its important
            self.do_dh_exchange(peerSock, initiating)
            
            # Start a thread to receive messages in parallel
            receiver_thread = threading.Thread(target=self.recv_msg,
                                               args=(peerSock,),
                                               daemon=True)
            receiver_thread.start()
            
            print("\nChat starting! Type 'EOM' to end ing chat.")
            
            while self.chat_active:
                try:
                    user_input = input("[Send]: ")
                    if not self.transmit_msg(peerSock, user_input):
                        break
                except EOFError:
                    break
            
        except Exception as e:
            print(f"Chat error: {e}")
        finally:
            self.chat_active = False
            peerSock.close()
            print("\nChat session ended")

    # This method is receiving messages from peer, dont worry about logic
    def recv_msg(self, peerSock):
        while self.chat_active:
            try:
                recvd = peerSock.recv(1024).decode()
                if not recvd:
                    break
                if recvd == "EOM":
                    print("\nPeer ended ing chat")
                    self.chat_active = False
                    break
                
                print(f"\n[Cipher]: {recvd}")
                # decrypt the received message using our aes key
                plain_text = CryptUtil.decrypt_message(self.aes_secret, recvd)
                if plain_text:
                    print(f"[Plain]: {plain_text}")
                print("[Send]: ", end='', flush=True)
                
            except Exception as ex:
                if self.chat_active:
                    print(f"\nError receiving msg: {ex}")
                break

    # This thread accept incoming chat connection, dont forget
    def accept_incoming(self):
        while self.keep_running:
            try:
                peerSock, _ = self.listenSock.accept()
                # if already in a chat then refuse connection
                if self.chat_active:
                    peerSock.close()
                    continue
                # create a new thread to handle the chat session
                chat_thread = threading.Thread(target=self.chat_session,
                                               args=(peerSock, False),
                                               daemon=True)
                chat_thread.start()
            except Exception:
                break

    # This method connect to a peer given its user id
    def connect_peer(self, target_user):
        if target_user not in self.user_list:
            print(f"User {target_user} not available")
            return
        if self.chat_active:
            print("Already in ing a chat session")
            return
            
        peer_ip, peer_port = self.user_list[target_user]
        peerSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            peerSock.connect((peer_ip, peer_port))
            self.chat_session(peerSock, True)
        except Exception as ex:
            print(f"Connection error: {ex}")
            peerSock.close()

    # This function refreshes the user list from server, dont use other libs
    def refresh_user_list(self):
        self.srvSock.send("GET_USERS".encode())
        self.user_list = json.loads(self.srvSock.recv(1024).decode())
        return self.user_list

    def run_client(self):
        try:
            while self.keep_running:
                if not self.chat_active:
                    print("\nOptions:")
                    print("1. List users")
                    print("2. Connect to a user")
                    print("3. Exit")
                    choice = input("Choice: ")
                    
                    if choice == "1":
                        users = self.refresh_user_list()
                        print("\nCurrent connected users:")
                        for uid in users:
                            if uid != self.user_id:
                                print(f"- {uid}")
                    elif choice == "2":
                        target = input("Enter user name to connect with him/her: ")
                        self.connect_peer(target)
                    elif choice == "3":
                        break
                else:
                    # Wait a moment if user is already in chating
                    threading.Event().wait(1)
                    
        finally:
            self.keep_running = False
            self.srvSock.close()
            self.listenSock.close()

if __name__ == "__main__":
    user_input = input("Enter your userID: ")
    client_inst = SecChatClient(user_input)
    client_inst.run_client()


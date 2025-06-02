# client.py - an secure chat program 4 talking to other peeps!!
import socket
import threading
import json
import os
import sys
from dh_utils import KeyMaker
from crypto_utils import MessageScrambler

class ChatBuddy:
    def __init__(self, nickname, server_addr='192.168.155.111', server_port=8080):
        # gettin DH numbers from environment innit
        self.big_prime = int(os.getenv('P'))
        self.base_num = int(os.getenv('Q'))
        
        # settin up basic stuff
        self.nickname = nickname
        self.server_spot = server_addr
        self.server_door = server_port
        self.still_running = True
        self.chatting_now = False
        self.chat_socket = None
        self.chat_signal = threading.Event()
        
        # makin socket 4 listening to other peeps
        self.ear_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ear_socket.bind(('', 0))
        self.my_door = self.ear_socket.getsockname()[1]
        self.ear_socket.listen(5)  # can handle 5 connection requests
        
        # connecting 2 main server
        self.server_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_link.connect((server_addr, server_port))
        self.server_link.send(f"{nickname}:{self.my_door}".encode())
        
        # getting who else is here
        self.online_peeps = json.loads(self.server_link.recv(1024).decode())
        
        # starting 2 listen 4 new friends
        self.doorbell = threading.Thread(target=self.wait_4_friends, daemon=True)
        self.doorbell.start()

    def do_key_exchange(self, friend_socket, im_first):
        """doin the diffie-hellman dance with friend"""
        try:
            my_secret = KeyMaker.make_secret_num()
            my_public = KeyMaker.calc_public_num(my_secret, self.base_num, self.big_prime)
            
            if im_first:
                # sending my public key first
                print(f"[Send]: Public Key = {my_public}")
                friend_socket.send(str(my_public).encode())
                their_public = int(friend_socket.recv(1024).decode())
                print(f"[Received]: Public Key = {their_public}")
            else:
                # getting their key first
                their_public = int(friend_socket.recv(1024).decode())
                print(f"[Send]: Public Key = {their_public}")
                print(f"[Received]: Public Key = {my_public}")
                friend_socket.send(str(my_public).encode())
            
            # making final secret n key
            shared_secret = KeyMaker.calc_shared_secret(my_secret, their_public, self.big_prime)
            secret_key = KeyMaker.make_aes_key(shared_secret)
            print(f"AES Key: {secret_key}")
            return secret_key
        except Exception as whoopsie:
            print(f"OOps, key exchange failed: {whoopsie}")
            return None

    def send_stuffs(self, socket_conn, secret_key):
        """sending messages to friend"""
        try:
            while self.chatting_now:
                msg = input()
                if msg == "EOM":
                    socket_conn.send(msg.encode())
                    self.chatting_now = False
                    break
                
                scrambled = MessageScrambler.scramble_msg(secret_key, msg)
                if scrambled:
                    socket_conn.send(scrambled.encode())
                print("[Send]: ", end='', flush=True)
        except Exception as whoopsie:
            print(f"\nUh oh, sending failed: {whoopsie}")
            self.chatting_now = False

    def get_stuffs(self, socket_conn, secret_key):
        """getting messages from friend"""
        try:
            while self.chatting_now:
                msg = socket_conn.recv(1024).decode()
                if not msg:
                    break
                if msg == "EOM":
                    print("\nFriend is lefting the chat")
                    self.chatting_now = False
                    break
                
                plain_msg = MessageScrambler.unscramble_msg(secret_key, msg)
                if plain_msg:
                    print(f"\n[Received]: {plain_msg}")
                    print("[Send]: ", end='', flush=True)
        except Exception as whoopsie:
            if self.chatting_now:
                print(f"\nOOps, receiving failed: {whoopsie}")
            self.chatting_now = False

    def chat_with_friend(self, friend_socket, im_first):
        """handling an whole chat session innit"""
        try:
            self.chat_socket = friend_socket
            self.chatting_now = True
            
            # doing key exchange first
            secret_key = self.do_key_exchange(friend_socket, im_first)
            if not secret_key:
                return
            
            print("\nSecure chat ready! Type 'EOM' to terminate the chat")
            print("[Send]: ", end='', flush=True)
            
            # starting sender n receiver
            sender = threading.Thread(target=self.send_stuffs, args=(friend_socket, secret_key))
            receiver = threading.Thread(target=self.get_stuffs, args=(friend_socket, secret_key))
            
            sender.daemon = True
            receiver.daemon = True
            
            sender.start()
            receiver.start()
            
            while self.chatting_now:
                sender.join(0.1)
                receiver.join(0.1)
                if not sender.is_alive() and not receiver.is_alive():
                    break
            
        except Exception as whoopsie:
            print(f"Chat went wrong: {whoopsie}")
        finally:
            self.chatting_now = False
            self.chat_socket = None
            friend_socket.close()

    def wait_4_friends(self):
        """waiting 4 other peeps to connect"""
        while self.still_running:
            try:
                friend_socket, _ = self.ear_socket.accept()
                if not self.chatting_now:
                    self.chat_with_friend(friend_socket, False)
                else:
                    friend_socket.close()
            except Exception as whoopsie:
                if self.still_running:
                    print(f"Connection failed: {whoopsie}")

    def call_friend(self, friend_name):
        """start chat with someone"""
        if self.chatting_now:
            print("OOps you'r already chatting!")
            return
            
        if friend_name not in self.online_peeps:
            print(f"Can't find {friend_name} :(")
            return
            
        friend_ip, friend_door = self.online_peeps[friend_name]
        friend_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            friend_socket.connect((friend_ip, friend_door))
            self.chat_with_friend(friend_socket, True)
        except Exception as whoopsie:
            print(f"Failed 2 connect: {whoopsie}")
            friend_socket.close()

    def get_whos_online(self):
        """checking who's around"""
        self.server_link.send("GET_USERS".encode())
        self.online_peeps = json.loads(self.server_link.recv(1024).decode())
        return self.online_peeps

    def start_chatting(self):
        """main program loop innit"""
        try:
            while True:
                if not self.chatting_now:
                    print("\nSelect one option which you want to do?")
                    print("1. See who's here")
                    print("2. Chat with someone")
                    print("3. Exit!")
                    choice = input("Pick one: ")
                    
                    if choice == "1":
                        peeps = self.get_whos_online()
                        print("\nPeeps online:")
                        for peep in peeps:
                            if peep != self.nickname:
                                print(f"- {peep}")
                    elif choice == "2":
                        friend = input("Who u wanna chat with? ")
                        self.call_friend(friend)
                    elif choice == "3":
                        break
                else:
                    self.chat_signal.wait(1)
        finally:
            self.still_running = False
            if self.chat_socket:
                self.chat_socket.close()
            self.server_link.close()
            self.ear_socket.close()

if __name__ == "__main__":
    nickname = input("What's ur name? ")
    buddy = ChatBuddy(nickname)
    buddy.start_chatting()

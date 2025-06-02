# server.py - making an chat server thats handling multiple users n stuff!!
import socket
import threading
import json

class ChatroomServer:   # dis class manages all da chatting stuff
    def __init__(self, port=8080):
        # initialising stuff for r server
        self.listenin_port = port
        self.active_peeps = {}  # storing who's currently chilling here
        self.peeps_mutex = threading.Lock()  # makin sure no1 mess with user list at same time
        
        # making an socket 4 listening to new connections n stuff
        self.main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # let us reuse da port if its already used b4
        self.main_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.main_socket.bind(('192.168.155.111', port))  # binding 2 all interfaces
        self.main_socket.listen(100)  # can handle upto 100 ppl chatting!!
        
        print(f"Yo! Server is now running on port {port}...")

    def deal_with_user(self, user_socket, address):
        """handling each connected user in they own thread innit"""
        try:
            # getting username n port where they listenin at
            incoming_data = user_socket.recv(1024).decode()
            nickname, listening_port = incoming_data.split(':')
            listening_port = int(listening_port)
            
            # putting user info in r dictionary but carefully
            with self.peeps_mutex:
                self.active_peeps[nickname] = (address[0], listening_port)
                print(f"New user joined: {nickname} with ip address and port {address[0]}:{listening_port}")
                
                # sending current user list 2 new person
                user_socket.send(json.dumps(self.active_peeps).encode())
            
            # keeping connection alive n handling requests
            while True:
                try:
                    msg = user_socket.recv(1024).decode()
                    if not msg:  # connection ded
                        break
                    
                    # if they want list of users, send it
                    if msg == "GET_USERS":
                        with self.peeps_mutex:
                            user_socket.send(json.dumps(self.active_peeps).encode())
                except:
                    break  # sumthing went wrong, bail out
                    
        except Exception as whoopsie:
            print(f"OOps, error with {address}: {whoopsie}")
            
        finally:
            # cleaning up when user leaves
            with self.peeps_mutex:
                for nick, (ip, port) in list(self.active_peeps.items()):
                    if ip == address[0] and port == listening_port:
                        del self.active_peeps[nick]
                        print(f"Seeaa! {nick} left the chat")
                        break
            user_socket.close()

    def run_forever(self):
        """keeps server running n accepting new connections till shutdown"""
        try:
            while True:
                new_socket, new_addr = self.main_socket.accept()
                # makin new thread 4 each connection
                new_thread = threading.Thread(target=self.deal_with_user,
                                           args=(new_socket, new_addr))
                new_thread.daemon = True
                new_thread.start()
        except KeyboardInterrupt:
            print("\n(Server Shutting down)...")
        finally:
            self.main_socket.close()

if __name__ == "__main__":
    chatroom = ChatroomServer()
    chatroom.run_forever()

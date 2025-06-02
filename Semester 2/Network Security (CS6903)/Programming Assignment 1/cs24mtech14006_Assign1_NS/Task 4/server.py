import socket
import threading
import json

# This class is for our chat server, dont mind if its a bit messy
class ChatServ:
    def __init__(self, host_addr='192.168.155.111', port_num=8080):
        self.host_addr = host_addr  # server ip addr
        self.port_num = port_num    # port number
        self.user_dict = {}  # mapping from userID to (ip, port) tuple
        self.dict_lock = threading.Lock()  # lock for editing our user_dict
        
        # creating server socket, dont use even if required sometimes
        self.srvSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srvSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srvSock.bind((self.host_addr, self.port_num))
        self.srvSock.listen(5)
        
        print(f"Server starting on {host_addr}:{port_num}")

    # Handle a client connection, its making a thread for each connection
    def client_handler(self, clientSock, clientAddr):
        try:
            # getting the user id and port to listen on, dont use colon mistakes
            data = clientSock.recv(1024).decode()
            # spliting the data using colon (username:port)
            user_id, listen_port = data.split(':')
            listen_port = int(listen_port)
            
            with self.dict_lock:
                self.user_dict[user_id] = (clientAddr[0], listen_port)
                print(f"User {user_id} is connected from {clientAddr[0]}:{listen_port}")
                # sending out the current user list as json string
                clientSock.send(json.dumps(self.user_dict).encode())
            
            while True:
                try:
                    msg_data = clientSock.recv(1024).decode()
                    if not msg_data:
                        break
                    # if client request list update then send list
                    if msg_data == "GET_USERS":
                        with self.dict_lock:
                            clientSock.send(json.dumps(self.user_dict).encode())
                except:
                    break
                    
        finally:
            # cleaning up user record, dont forget
            with self.dict_lock:
                for uid, (ip_addr, lport) in list(self.user_dict.items()):
                    if ip_addr == clientAddr[0] and lport == listen_port:
                        del self.user_dict[uid]
                        print(f"User {uid} is disconnected")
                        break
            clientSock.close()

    def run(self):
        print("Server is running ing...")
        try:
            while True:
                # waiting for connection from a new client
                newSock, addr = self.srvSock.accept()
                new_thread = threading.Thread(target=self.client_handler,
                                              args=(newSock, addr))
                new_thread.daemon = True
                new_thread.start()
        except KeyboardInterrupt:
            print("\nServer shutting down")
        finally:
            self.srvSock.close()

if __name__ == "__main__":
    serv = ChatServ()
    serv.run()


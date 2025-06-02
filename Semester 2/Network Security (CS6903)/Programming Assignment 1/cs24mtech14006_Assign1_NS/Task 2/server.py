# NS assignment 1
# Task -2
# By-
# Anurag Sarva
# Gulshan Hatzade

# Server Code

# CentralHub - The main server program for our chat system
import socket  # For network communications
import threading  # For handling multiple users at once
import sys  # For system-level operations

# Main configuration settings
WELCOME_PORT = 8080  # Port where we accept new connections
MAXIMUM_USERS = 100  # How many people can connect at once

class PersonInfo:
    # Storing information about each connected person
    def __init__(self, location, doorway, connection, displayname=""):
        self.location = location  # Their IP address
        self.doorway = doorway  # Their port number
        self.connection = connection  # Their active connection
        self.displayname = displayname  # Their chosen username

class CentralHub:
    def __init__(self):
        # Initialize our server setup
        self.connected_people = []  # List to keep track of everyone
        self.safety_lock = threading.Lock()  # For safe list modifications
        
        # Creating main server connection point
        self.main_doorway = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allowing port reuse for easier restarts
        self.main_doorway.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # Starting up the server
            self.main_doorway.bind(('192.168.198.111', WELCOME_PORT))
            self.main_doorway.listen(MAXIMUM_USERS)
            print(f"Server is now running!")
            print(f"Accepting connections on port {WELCOME_PORT}...")
            print(f"Can handle up to {MAXIMUM_USERS} simultaneous users\n")
        except Exception as problem:
            print(f"Startup failed: {problem}")
            sys.exit(1)

    def remove_person(self, their_connection):
        # Safely removing someone when they disconnect
        with self.safety_lock:
            self.connected_people = [p for p in self.connected_people if p.connection != their_connection]

    def handle_person(self, person_info):
        # Managing individual connection
        their_connection = person_info.connection
        
        try:
            # Getting their chosen username
            displayname = their_connection.recv(1024).decode().strip()
            person_info.displayname = displayname
            print(f"New user '{displayname}' has joined.")

            # Main loop for handling their requests
            while True:
                request_type = their_connection.recv(1024).decode().strip()
                if not request_type:
                    break
                if request_type == "LIST":
                    # Creating list of who's online
                    response = "=== Currently Online Users ===\n"
                    with self.safety_lock:
                        for person in self.connected_people:
                            response += f"Username: {person.displayname}\n"
                            response += f"Address: {person.location}\n"
                            # response += f"Port: {person.doorway}\n"
                            response += "-------------------\n"
                    their_connection.send(response.encode())

        except Exception as problem:
            print(f"Problem with user {person_info.displayname}: {problem}")
        finally:
            # Cleanup when they disconnect
            print(f"User '{person_info.displayname}' has disconnected.")
            self.remove_person(their_connection)
            their_connection.close()

    def start_serving(self):
        # Main server loop
        while True:
            try:
                # Waiting for new connections
                new_connection, address = self.main_doorway.accept()
                new_person = PersonInfo(address[0], address[1], new_connection)
                
                # Safely adding new person to our list
                with self.safety_lock:
                    self.connected_people.append(new_person)
                
                print(f"New connection from {address[0]}:{address[1]}")
                
                # Creating separate thread for this person
                person_handler = threading.Thread(target=self.handle_person, args=(new_person,))
                person_handler.daemon = True
                person_handler.start()
                
            except KeyboardInterrupt:
                # Handling clean shutdown
                print("\nServer shutting down...")
                break
            except Exception as problem:
                print(f"Connection error: {problem}")

        # Cleanup when server shuts down
        self.main_doorway.close()

if __name__ == "__main__":
    # Starting point of server program
    central = CentralHub()
    central.start_serving()
# NS assignment 1
# Task -2
# By-
# Anurag Sarva
# Gulshan Hatzade

# Client Code

# ChatBuddy - A friendly program for connecting people
import socket  # Using this to make network connections happen
import threading  # Helps us do multiple things at once
import sys  # Need this for some system operations

# These are our main connection settings
CENTRAL_ADDRESS = "192.168.198.111"  # The address where our main server lives
CENTRAL_DOORWAY = 8080  # The port number our server uses to welcome people
LOCAL_DOORWAY = 9090  # Where we'll be listening for incoming chats

def display_message(message_content):
    # Making sure messages show up right away using flush
    print(message_content, flush=True)

class MessageBuddy:
    def __init__(self):
        # Setting up initial values when someone starts the program
        self.person_name = ""  # Going to store the user's name here
        self.main_link = None  # This will be our connection to the server

    def receive_incoming(self, active_connection):
        # This keeps checking for new messages from our chat partner
        while True:
            try:
                # Getting whatever message comes through
                incoming_words = active_connection.recv(1024).decode()
                if not incoming_words:
                    # If we get nothing, means friend probably left
                    display_message("\nLooks like your friend stepped away.")
                    break
                # Showing what we got and prompting for response
                display_message(f"\n[Received]: {incoming_words}")
                display_message("[Send]: ")
            except:
                # Something went wrong with the connection
                break

    def start_conversation(self, friend_address, friend_doorway):
        # Beginning the process of connecting to another person
        display_message(f"\nTrying to reach your friend at {friend_address}:{friend_doorway}...")
        
        try:
            # Creating new connection specifically for this chat
            friend_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            friend_link.connect((friend_address, friend_doorway))
            
            # Letting user know we made it through
            display_message("Connection successful! Chat away!")
            display_message("When you're done, just type 'EOM'\n")
            
            # Setting up message receiving in background
            message_watcher = threading.Thread(target=self.receive_incoming, args=(friend_link,))
            message_watcher.daemon = True  # This makes it close when main program ends
            message_watcher.start()
            
            # Main chat loop where we send messages
            while True:
                display_message("[Send]: ")
                outgoing_words = input()
                friend_link.send(outgoing_words.encode())
                if outgoing_words == "EOM":
                    # User wants to end chat
                    break
                    
            friend_link.close()  # Cleaning up connection when done
            
        except Exception as problem:
            display_message(f"Something went wrong with the chat: {problem}")

    def wait_for_friends(self):
        # Setting up to receive incoming chat requests
        display_message("\nWaiting for someone to start a chat...")
        
        try:
            # Creating listening point for incoming connections
            waiting_spot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            waiting_spot.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            waiting_spot.bind(('', LOCAL_DOORWAY))
            waiting_spot.listen(1)
            
            display_message(f"Ready for chat on port {LOCAL_DOORWAY}...")
            
            # Accepting incoming connection when it happens
            new_friend, friend_location = waiting_spot.accept()
            display_message("Someone joined! Start chatting!")
            display_message("Remember to type 'EOM' when you're finished\n")
            
            # Setting up message receiving
            message_watcher = threading.Thread(target=self.receive_incoming, args=(new_friend,))
            message_watcher.daemon = True
            message_watcher.start()
            
            # Main chat loop
            while True:
                display_message("[Send]: ")
                outgoing_words = input()
                new_friend.send(outgoing_words.encode())
                if outgoing_words == "EOM":
                    break
            
            # Cleaning up connections
            new_friend.close()
            waiting_spot.close()
            
        except Exception as problem:
            display_message(f"Something went wrong while waiting: {problem}")

    def get_online_friends(self):
        # Asking server who else is around
        display_message("\nChecking who's online...\n")
        self.main_link.send("LIST".encode())
        server_response = self.main_link.recv(1024).decode()
        display_message(server_response)

    def begin_program(self):
        # Main program startup sequence
        display_message("Getting everything ready...\n")
        
        try:
            # Connecting to main server
            self.main_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.main_link.connect((CENTRAL_ADDRESS, CENTRAL_DOORWAY))
            display_message("Successfully connected to main server!\n")
            
            # Getting user's name
            display_message("What should we call you?: ")
            self.person_name = input()
            self.main_link.send(self.person_name.encode())
            display_message("Name registered successfully.")
            
            # Main program loop
            while True:
                display_message("\nWhat would you like to do?")
                display_message("1. See who's online")
                display_message("2. Start a chat with someone")
                display_message("3. Wait for someone to chat")
                display_message("4. Exit program")
                display_message("Choose your option (1-4): ")
                
                user_choice = input()
                
                # Handling user's choice
                if user_choice == "1":
                    self.get_online_friends()
                elif user_choice == "2":
                    print("Use this port to coonet with the peer ")
                    print(LOCAL_DOORWAY)
                    display_message("Enter their address: ")
                    their_place = input()
                    display_message("Enter their port: ")
                    their_port = int(input())
                    self.start_conversation(their_place, their_port)
                elif user_choice == "3":
                    self.wait_for_friends()
                elif user_choice == "4":
                    display_message("Thanks for using ChatBuddy!")
                    break
                else:
                    display_message("That's not a valid option, try again")
                    
        except Exception as problem:
            # Catching any unexpected problems
            display_message(f"An error occurred: {problem}")
        finally:
            # Making sure we clean up our connection
            if self.main_link:
                self.main_link.close()

if __name__ == "__main__":
    # Starting point of our program
    chat_program = MessageBuddy()
    chat_program.begin_program()
import socket

def run_client():
    # create socket for client
    user_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # connect to math server
        user_socket.connect(('127.0.0.1', 8080))
        
        while True:
            try:
                print("Enter expression here...")
                # get math problem from user
                math_problem = input().strip()
                
                # send problem to server
                user_socket.send(math_problem.encode())
                
                # check if user wants to quit
                if math_problem == 'END':
                    break
                    
                # get answer from server
                answer = user_socket.recv(1024).decode()
                # check if there was an error
                if answer.startswith('Error:'):
                    # print(answer)
                    print("Error, you entered wrong formate of expression.")
                else:
                    # show the result
                    print(f"RESULT: {answer}")
                    
            except EOFError:
                break
                
    finally:
        # close connection
        user_socket.close()

if __name__ == "__main__":
    run_client()

import socket
from calculator import Calculator

def run_server():
    # make new server socket
    math_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # allow reusing address
    math_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # set server location
    math_server.bind(('127.0.0.1', 8080))
    # wait for connections
    math_server.listen(1)
    
    print("Server ready... waiting for anyone to connect...")
    
    # accept incoming connection
    user_socket, user_address = math_server.accept()
    print(f"Got connection from {user_address}")
    
    # create calculator object
    math_helper = Calculator()
    
    try:
        while True:
            # get expression from user
            incoming = user_socket.recv(1024).decode()
            # check if user wants to quit
            if not incoming or incoming.strip() == 'END':
                break
                
            try:
                # break down expression into parts
                expression_parts = math_helper.parse_expression(incoming)
                # calculate the answer
                answer = math_helper.evaluate_expression(expression_parts)
                
                # send answer back to user
                user_socket.send(str(answer).encode())
            except Exception as problem:
                # handle any errors
                error_text = f"Error: {str(problem)}"
                user_socket.send(error_text.encode())
                
    finally:
        # close all connections
        user_socket.close()
        math_server.close()

if __name__ == "__main__":
    run_server()

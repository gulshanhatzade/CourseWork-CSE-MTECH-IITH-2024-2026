Programming Assignment 1 CS6903: Network Security


Task 1: Client-Server Calculator

This task implements a basic client-server model where the client sends mathematical expressions to the server, and the server evaluates them following BODMAS rules. The implementation consists of three files:

calculator.py: Implements a Calculator class to parse and evaluate expressions while maintaining operator precedence.

server.py: Implements a server that listens for client connections, processes incoming expressions using Calculator, and sends back results.

client.py: Implements a client that sends expressions to the server and receives results, displaying them in the specified format.

Execution Instructions

Step 1: Start the Server

    Open a terminal.

    Navigate to the directory containing server.py.

    Run the following command:

    python3 server.py

    The server will start and wait for client connections.

Step 2: Start the Client

    Open another terminal.

    Navigate to the directory containing client.py.

    Run the following command:

    python3 client.py

    The client will connect to the server.

Step 3: Sending Expressions

    Enter a mathematical expression in the client terminal. Example:

         1 + 5 * 2 - 1 + 10

    The server will process the expression and return the result:

    RESULT: 20

Step 4: Terminating the Client and Server

    To exit the client, type:

    END

    This will send a termination signal to the server, shutting it down properly.



Contribution of Group Members-

Anurag Sarva (CS24MTECH14003): Implemented calculator.py and logic for expression parsing and evaluation.

Gulshan Hatzade (CS24MTECH14006): Implemented server.py and client.py, handling socket communication and integration with the calculator.

ANTI-PLAGIARISM STATEMENT -

We certify that this assignment/report is our own work, based on our personal study and/or research
and that we have acknowledged all material and sources used in its preparation, whether books, articles,
packages, datasets, reports, lecture notes, or any other document, electronic or personal communication.
We also certify that this assignment/report has not previously been submitted for assessment/project in any
other course lab, except where specific permission has been granted from all course instructors involved, or
at any other time in this course, and that we have not copied in part or whole or otherwise plagiarized the
work of other students and/or persons. We pledge to uphold the principles of honesty and responsibility at
CSE@IITH. In addition, We understand my responsibility to report honor violations by other students if
we become aware of it.

Names:  Anurag Sarva, Gulshan Hatzade
Date:   04/02/2025
Signature: Anurag Sarva, Gulshan Hatzade




# CS6013 Assignment 2

**Roll Number:** CS24MTECH14006

## Contents
- [Assignment Description](#assignment-description)
- [Problem 1: Maximum Length Palindrome in a String](#problem-1-maximum-length-palindrome-in-a-string)
- [Problem 2: Railway Station Placement Optimization](#problem-2-railway-station-placement-optimization)
- [Requirements](#requirements)
- [Compilation and Execution Instructions](#compilation-and-execution-instructions)
- [Code Overview](#code-overview)
- [External References](#external-references)

## Assignment Description

This assignment includes two tasks by using linked list, one task is of dynamic programming & the other task is on greedy algorithms in C. Each problem demands linked list data handling and robust input validation.

### Problem 1: Maximum Length Palindrome in a String
Design the dynamic programming algorithm for indentifying longest palindromic subsequence in an input string.

**Input:** A string entered involving charters from (a to z) or (A to Z)  
**Output:** The length & the characters of the longest palindromic subsequence, or an error message if input is invalid.

### Problem 2: Railway Station Placement Optimization
Design the greedy algorithm for minimizing number of railway stations required to cover towns along a line, given a distance threshold `d`.

**Input:** A sorted sequence of towns as a linked list, and the maximum allowable distance, `d`, between the town and its nearest station.  
**Output:** Optimal station placement to cover all towns or an error message if input is invalid.

## Requirements

- **Language:** C
- **Data Structures:** Linked lists for all data representation
- **Error Handling:** Each program must validate inputs & show error messages for invalid cases.
- **Documentation:** Code should be documented with explanations of each function, parameter & the return values.

## Compilation and Execution Instructions

1. **Compile the Programs**  
   Compile each problem file separately using `gcc`.

   ```bash
   gcc problem1_cs24mtech14006.c -o problem1_cs24mtech14006
   gcc problem2_cs24mtech14006.c -o problem2_cs24mtech14006

2. **Run the Compiled Programs**

   ### Problem 1: Finding Longest Palindromic Subsequence
   - Execute the command
     ```bash
     ./problem1_cs24mtech14006
     ```
   - **Input:** Enter a string of alphabetic characters (A-Z, a-z) when prompted, then press Enter when string is completed.
   ```bash
   Enter input characters one by one (press Enter to stop):
    ```
    When this appears, enter the string and press enter
    ```bash
   Enter input characters one by one (press Enter to stop):
   badsasdar
    ```
   - **Output:** It will display longest palindromic subsequence & its length. An error message is shown if the input contains non-alphabetic characters.
   ```bash
   Enter input characters one by one (press Enter to stop):
   badsasdar
   Input linked list isl badsasdar
   Longest palindromic subsequence is- adsasda
   Length of the subsequence is  7
    ```


   ### Problem 2: Optimizing Railway Station Placement
   - Execute:
     ```bash
     ./problem2_cs24mtech14006
   
   - **Input:** Enter the sorted sequence of town locations followed by  distance `d` when prompted. Locations should represent towns on a number line.
   First enter number of towns and press enter
   ```bash
   Enter number of towns:- 5
    ``` 
    Now enter the tolerance distance here and press enter
    ```bash
   Enter number of towns:- 5
   Enter distance tolerance d:- 3
    ``` 
    Now enter the locations of towns, after entering each town position press enter
    ```bash
   Enter number of towns:- 5
   Enter distance tolerance d:- 3
   Enter positions of towns in sorted order (press Enter after each town postition):-
   4
   7
   9
   22
   33
    ``` 

   
   - **Output:** Shows greedy placement of stations for full town coverage within distance `d`, or an error if the input is invalid.
   ```bash
   Optimal placement of stations is as follows:-
   Station 1 is at position 7.
   Station 2 is at position 25.
   Station 3 is at position 36.

   Total stations needed are 3.
    ``` 

## Code Overview

### Problem 1: Maximum Length Palindromic Subsequence

1. **Linked List Structure (`Node`):** 
   - Each node holds single character & pointer to the next node.
   
2. **Functions Overview:**
   - **`create_node(char value)`:** Creates the new node with character.
   - **`check_input_is_valid(Node* head)`:** Validates that only alphabetic characters are in linked list.
   - **`find_length(Node* head)`:** Returns length of linked list.
   - **`get_node_at(Node* head, int index)`:** Retrieves node at the specific index.
   - **`find_longest_palindromic_subsequence(Node* head)`:** Implements dynamic programming for dinding longest palindromic subsequence.
   - **`create_linked_list_from_input()`:** Builds the linked list from user input.
   - **`free_linked_list(Node* head)`:** Frees linked list memory to avoid leaks.

3. **Dynamic Programming Approach:**
   - The 2D array (`dp`) stores results for subproblems, where `dp[i][j]` represents the length of longest palindromic subsequence within the substring from `i` to `j`.
   - The function retrieves & displays the palindrome based on this DP table.

### Problem 2: Railway Station Placement Optimization

1. **Linked List Structure (`TownNode`):** 
   - Each node holds an integer town location & pointer to the next town.
   
2. **Functions Overview:**
   - **`create_town_node(int location)`:** Creates node for each town location.
   - **`check_input_is_sorted(TownNode* head)`:** Validates that towns are sorted in ascending order.
   - **`place_stations(TownNode* head, int d)`:** Uses the greedy algorithm to determine minimal station placement.
   - **`create_linked_list_from_town_input()`:** Builds the linked list from user-entered town locations.
   - **`free_town_linked_list(TownNode* head)`:** Frees linked list memory to avoid leaks.

3. **Greedy Approach:**
   - Starts from  first town and places a station as far as possible within distance `d`.
   - Continues through the list, ensuring each town is covered with the fewest number of stations.

## External References

- **Dynamic Programming for Palindromic Subsequence:** [Longest Palindromic Subsequence](http://users.ece.northwestern.edu/~dda902/336/hw6-sol.pdf)
- **Greedy Algorithm Basics:** [Greedy Railway station allotment Algorithm](https://cs.colby.edu/courses/F21/cs375/375ProblemSet1Sol.pdf)


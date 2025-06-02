#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

// ============================= Linked List Node Structure =============================

/**
 * This is structure for representing each node of linked list.
 * Here each node is containing the character value the pointer which is pointing to next node.
 */
typedef struct Node {
    char value;         // this is for character stored in node
    struct Node* next;  // this is for ointer to next node in linked list
} Node;



// =========================== Linked List Helper Functions ===========================

/**
 * This is Function for creating the new node with the given character value.
 *
 * Inputs - - - value character to be stored in new node.
 * Returns - - -  pointer to newly created node.
 */
Node* create_node(char value) {
    Node* new_node = (Node*)malloc(sizeof(Node)); // Allocating memory for new node here by using malloc
    new_node->value = value;  // Setting value of node
    new_node->next = NULL;    // Initializing next as NULL
    return new_node;
}

/**
 * This is the Function for checking if input string in linked list is valid.
 * Only alphabetic characters (a-z, A-Z) are considered valid.
 *
 * Inputs - - - head Pointer to head of linked list.
 * Returns - - - 1 if all characters are valid otherwise 0 if any invalid character is found.
 */
int check_input_is_valid(Node* head) {
    Node* current_node = head;
    while (current_node != NULL) {
        if (!isalpha(current_node->value)) {
            return 0; // Returning 0 when an invalid character is found
        }
        current_node = current_node->next;
    }
    return 1; // Returning 1 for valid input
}

/**
 * This is the Function for finding length of linked list.
 *
 * Inputs - - - head Pointer to head of linked list.
 * Returns - - - length of linked list as an integer.
 */
int find_length(Node* head) {
    int length = 0;
    Node* current_node = head;
    while (current_node != NULL) {
        length++;                // Incrementing length for each node
        current_node = current_node->next;
    }
    return length;
}

/**
 * This Function is playing role for getting node at specific index in linked list.
 *
 * Inputs - - - head Pointer to head of linked list.
 * Inputs - - - index index of node to retrieve (by following 0 based index).
 * Returns - - - The pointer to node at specified index.
 */
Node* get_node_at(Node* head, int index) {
    Node* current_node = head;
    for (int i = 0; i < index; i++) {
        current_node = current_node->next; // Traversing to desired index
    }
    return current_node;
}

// ========================== Dynamic Programming Function ============================

/**
 * Function for finding longest palindromic subsequence using the dynamic programming approach.
 * It prints longest palindromic subsequence & its length.
 *
 * Inputs - - - head Pointer to head of linked list.
 */
void find_longest_palindromic_subsequence(Node* head) {
    // Checking if input string is valid
    if (!check_input_is_valid(head)) {
        printf("Error- Input is not valid. Only letters (a-z, A-Z) are allowed.\n");
        return;
    }

    // Finding length of linked list
    int n = find_length(head);
    if (n == 0) {
        printf("Error- Input string is empty.\n");
        return;
    }

    // Creating 2D array for DP table
    int** dp = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dp[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            dp[i][j] = (i == j) ? 1 : 0; // Single character is a palindrome of length 1
        }
    }

    // Filling DP table for subsequences of length 2 or more
    for (int length = 2; length <= n; length++) {
        for (int i = 0; i <= n - length; i++) {
            int j = i + length - 1; // Ending index of current subsequence
            if (get_node_at(head, i)->value == get_node_at(head, j)->value) {
                dp[i][j] = dp[i + 1][j - 1] + 2; // Characters matched here, then expanding palindrome
            } else {
                dp[i][j] = (dp[i + 1][j] > dp[i][j - 1]) ? dp[i + 1][j] : dp[i][j - 1]; // Choose maximum
            }
        }
    }

    // Retrieving longest palindromic subsequence from DP table
    int max_length = dp[0][n - 1]; // maximum length is stored in dp[0][n-1]
    char* result = (char*)malloc((max_length + 1) * sizeof(char));
    result[max_length] = '\0'; // Null terminate result string
    int start = 0;
    int end = max_length - 1;
    int i = 0, j = n - 1;

    while (i <= j) {
        if (get_node_at(head, i)->value == get_node_at(head, j)->value) {
            result[start++] = get_node_at(head, i)->value; // Add matching character to start
            result[end--] = get_node_at(head, j)->value;   // Add matching character to end
            i++;
            j--;
        } else if (dp[i + 1][j] > dp[i][j - 1]) {
            i++; // Moveing down if dp[i+1][j] is greater
        } else {
            j--; // Moving left if dp[i][j-1] is greater
        }
    }

    // Printing result which are found in this problem
    printf("longest palindromic subsequence is: %s\n", result);
    printf("Length of subsequence: %d\n", max_length);

    // Freeing memory allocated for DP table & result
    for (int i = 0; i < n; i++) {
        free(dp[i]);
    }
    free(dp);
    free(result);
}

// ========================= Linked List Utility Functions =========================

/**
 * This is the Function for creating the linked list from user input characters.
 * It reads characters one by one until Enter key is pressed, after enter key scanning will be completed
 *
 * Returns - - - Pointer to head of created linked list.
 */
Node* create_linked_list_from_input() {
    Node* head_node = NULL;
    Node* tail_node = NULL;
    char input_char;

    printf("Enter input characters one by one (press Enter to stop):\n");
    while ((input_char = getchar()) != '\n') {
        if (input_char != EOF) {
            Node* new_node = create_node(input_char); // Creating new node for each character
            if (head_node == NULL) {
                head_node = new_node; // Initializing the head_node
                tail_node = new_node;
            } else {
                tail_node->next = new_node; // Now Appending to end of list
                tail_node = new_node;
            }
        }
    }

    return head_node;
}

/**
 * This is Function for freeing all nodes in the linked list for avoiding memory leaks.
 *
 * Inputs - - - head Pointer to head of linked list.
 */
void free_linked_list(Node* head) {
    Node* current_node = head;
    while (current_node != NULL) {
        Node* temp = current_node;
        current_node = current_node->next;
        free(temp); // Freeing current node
    }
}

// =============================== Main Function ================================

/**
 * This is the Main function to run program.
 * It will creates the linked list from user input & will finds longest palindromic subsequence,
 * & also frees allocated memory before exiting.
 */
int main() {
    Node* head = create_linked_list_from_input(); // Creating linked list from input

    printf("Input linked list: ");
    Node* current_node = head;
    while (current_node != NULL) {
        printf("%c", current_node->value);
        current_node = current_node->next;
    }
    printf("\n");

    find_longest_palindromic_subsequence(head);   // This is for work of Finding &  also printing longest palindromic subsequence

    free_linked_list(head); // Freeing the linked list
    return 0;
}

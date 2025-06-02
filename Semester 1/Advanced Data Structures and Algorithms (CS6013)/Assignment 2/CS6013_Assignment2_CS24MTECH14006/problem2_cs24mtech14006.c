#include <stdio.h>
#include <stdlib.h>

// ============================= Linked List Node Structure =============================

/**
 * This is the Structure for linked list nodes.
 * Each node represents the town or the station with its position on line.
 */
typedef struct Node {
    int position;           // This is for the Position of town or station on line
    struct Node* next;      // This is for Pointer for pointing to the next node in list
} Node;

// =========================== Linked List Helper Functions ===========================

/**
 * This is the Function for creating new node with the given position.
 *
 * Input - - - position position of town or station as per the requirement
 * Returns - - - The pointer pointing to the newly created node.
 */
Node* create_node(int position) {
    Node* new_node = (Node*)malloc(sizeof(Node)); // Allocating memory for new node
    new_node->position = position; // Setting position
    new_node->next = NULL; // Initializing next pointer to NULL
    return new_node; // Returning created node
}

/**
 * This is the Function for appending the node with the given position at end of linked list.
 *
 * Input - - - head Double pointer to the head of linked list.
 * Input - - - position position of new node to be added.
 */
void append_node_at_end(Node** head, int position) {
    Node* new_node = create_node(position); // This is Creating new node with given position
    if (*head == NULL) {
        *head = new_node; // If list is empty, setting new node as head
    } else {
        Node* temp = *head; // Traversing list for findin last node
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = new_node; // This will be Adding new node at end of list
    }
}

/**
 * This is the Function printing linked list of station positions.
 *
 * Input - - - It will take head Pointer pointing to head of linked list.
 */
void print_linked_list(Node* head) {
    int station_count = 1; // Counter for station numbering
    while (head != NULL) {
        printf("Station %d is at position %d.\n", station_count, head->position); // Printing station details
        station_count++; // Incrementing station number
        head = head->next; // This will be Moving to next node in list
    }
}

// ======================= Greedy Algorithm for Optimal Station Placement =======================

/**
 * This is Greedy algorithm for finding optimal placement of stations for covering all towns.
 * The station covers the town if it is within distance 'd' from town.
 *
 * Input - - - towns Pointer to linked list of towns positions (I assumed sorted).
 * Input - - - d - tolerance distance within which the station can cover the town.
 */
void find_optimal_stations_greedy(Node* towns, int d) {
    // Validating input
    if (towns == NULL || d <= 0) {
        printf("Input is invalid- no towns or invalid distance.\n");
        return; // Returning early if input is not valid
    }

    Node* stations = NULL;    // Linked list for storing optimal station positions
    Node* current_town = towns;  // Pointer for traversing towns list
    Node* last_station = NULL;   // Pointer for keeping track of last added station

    // Placing first station at first town's position + d
    append_node_at_end(&stations, current_town->position + d); // Adding first station
    last_station = stations; // Updating last_station to point to first station
    printf("Optimal placement of stations is as follows:-\n");
    printf("Station 1 is at position %d.\n", last_station->position); // Printing first station

    current_town = current_town->next; // Moving to next town
    int count_of_station = 1; // Starting station count from 1

    // Loop to place additional stations as needed
    while (current_town != NULL) {
        // Check if current town is covered by last station within distance 'd'
        if (abs(current_town->position - last_station->position) <= d) {
            // Town is already covered, so no need to place a new station
        } else {
            // Placing a new station at current town's position + d
            append_node_at_end(&stations, current_town->position + d);
            last_station = last_station->next; // Updating last_station to newly added station
            printf("Station %d is at position %d.\n", ++count_of_station, last_station->position); // Printing new station
        }
        current_town = current_town->next; // Moving to next town
    }

    // Printing total number of stations needed
    printf("\nTotal stations needed are %d.\n", count_of_station);
}

// ========================= Main Function =========================

/**
 * This is the Main function for taking user i/p, creating the linked list of towns, 
 * & finding optimal station placement using greedy algorithm.
 */
int main() {
    int n, d;

    // Taking input: number of towns
    printf("Enter number of towns:- ");
    scanf("%d", &n);

    // Checking if number of towns is valid
    if (n <= 0) {
        printf("Input is invalid- number of towns must be greater than 0.\n");
        return 1; // Return early for invalid input
    }

    // Taking input: distance tolerance
    printf("Enter distance tolerance d:- ");
    scanf("%d", &d);

    // Checking if distance tolerance is valid
    if (d <= 0) {
        printf("Input is invalid- distance tolerance must be greater than 0.\n");
        return 1; // Return early for invalid input
    }

    // Linked list for storing positions of towns
    Node* towns = NULL;
    printf("Enter positions of towns in sorted order (press Enter after each town postition):-\n");

    // Taking input: positions of towns (sorted) and storing in linked list
    for (int i = 0; i < n; i++) {
        int town_positions;
        scanf("%d", &town_positions);
        append_node_at_end(&towns, town_positions); // Appending town position to linked list
    }

    // Finding and printing optimal placement of stations
    find_optimal_stations_greedy(towns, d);

    return 0;
}

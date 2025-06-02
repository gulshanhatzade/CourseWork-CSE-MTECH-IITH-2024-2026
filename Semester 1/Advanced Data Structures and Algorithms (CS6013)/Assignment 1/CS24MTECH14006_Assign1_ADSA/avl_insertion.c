//AVL tree insertion
#include <stdio.h>
#include <stdlib.h>

//Structure of avl tree nodes
typedef struct Node {
    int data;   //value of node
    struct Node *l,*r;  //Pointers to child
    int height; //height of node
} Node;

Node *root = NULL;  //Global variable - root of avl tree

// Function for getting height of node in avl tree
int height(Node *N) {
    if (N == NULL)
        return 0;
    return N->height;
}

//Function for getting maximum of two integers
int maximum(int a, int b) {
    return (a > b) ? a : b;
}



// Function for performing right rotation around node y
Node* right_Rotate(Node *y, int *rotations) {
    Node *x = y->l;
    Node *T2 = x->r;

    // Perform right rotation
    x->r = y;
    y->l = T2;

    // Updating heights of y and x after rotation
    y->height = maximum(height(y->l), height(y->r)) + 1;
    x->height = maximum(height(x->l), height(x->r)) + 1;

    (*rotations)++; // Increment rotation count for tracking

    // Returning the new root
    return x;
}

// Function for performing left rotation around node x
Node* left_Rotate(Node *x, int *rotations) {
    Node *y = x->r;
    Node *T2 = y->l;

    // Perform left rotation
    y->l = x;
    x->r = T2;

    // Updating heights of x and y after rotation
    x->height = maximum(height(x->l), height(x->r)) + 1;
    y->height = maximum(height(y->l), height(y->r)) + 1;

    (*rotations)++; // Increment rotation count for tracking

    // Returning the new root
    return y;
}

// Function for creating new node with given data
Node* create_node(int data) {
    Node* new_node = (Node*)malloc(sizeof(Node));    //Allocating memory
    new_node->data = data;
    new_node->l = NULL;
    new_node->r = NULL;
    new_node->height = 1; // New nodes are initially added at leaf level, so height =1
    return new_node;
}

// Function for getting balance factor of a node
int get_balance(Node *N) {
    if (N == NULL)
        return 0;
    return height(N->l) - height(N->r); //Diff b/n heights of left and right subtrees
}









//Function for inseting new value in the AVL tree and rebalance
Node* insert_avl(Node* node, int data, int *rotations, int *track_tree_height) {
   // Step 1- Perform normal B S T insert_avlion
    if (node == NULL)
        return create_node(data);

    if (data < node->data)
        node->l = insert_avl(node->l, data, rotations, track_tree_height);
    else if (data > node->data)
        node->r = insert_avl(node->r, data, rotations, track_tree_height);
    else
        return node; // Duplicates are not allowed in  the AVL tree

    // Step 2- Updating height of current node
    node->height = 1 + maximum(height(node->l), height(node->r));

    // Step 3- Getting balance factor to check if the node is unbalanced
    int balance = get_balance(node);

    // Step 4- Rebalancing node if necessary by performing rotations
    if (balance > 1 && data < node->l->data)
        return right_Rotate(node, rotations);

    if (balance < -1 && data > node->r->data)
        return left_Rotate(node, rotations);

    if (balance > 1 && data > node->l->data) {
        node->l = left_Rotate(node->l, rotations);
        return right_Rotate(node, rotations);
    }

    if (balance < -1 && data < node->r->data) {
        node->r = right_Rotate(node->r, rotations);
        return left_Rotate(node, rotations);
    }

    return node;
}




// Function for saving AVL tree to file by using pre order traversal
void save_tree_to_file(Node *root, FILE *file) {
    if (root != NULL) {
        fprintf(file, "%d\n", root->data);
        save_tree_to_file(root->l, file);
        save_tree_to_file(root->r, file);
    }
}

// Function for freeing memory used by the AVL tree
void free_tree(Node *node) {
    if (node != NULL) {
        free_tree(node->l);
        free_tree(node->r);
        free(node);
    }
}

int main() {
    int sizes_array[] = {10000, 100000, 1000000, 10000000}; // Sizes of array of arrays

    for (int x = 0; x < 4; x++) {
        int array_size = sizes_array[x];
        float rotations_total = 0;
        float height_total = 0;

        // Process 100 different arrays, for each size
        for (int i = 1; i <= 100; i++) { 
            char filename[50];
            sprintf(filename, "array_size%d_%d.txt", array_size, i); // Create filename
            FILE *arrayFile = fopen(filename, "r");
            if (!arrayFile) {
                printf("Failed to open file- %s\n", filename);
                continue;
            }

            int value;
            int rotations = 0;
            int track_tree_height = 0;

            while (fscanf(arrayFile, "%d", &value) != EOF) {
                root = insert_avl(root, value, &rotations, &track_tree_height);
            }

            rotations_total += rotations;   // Accumulate rotations for current array
            height_total += height(root);   //Accumulate final height of the tree

            // Saving resulting tree structure to file
            char tree_file[50];
            sprintf(tree_file, "avl_tree_size%d_%d.txt", array_size, i);
            FILE *tree_filePtr = fopen(tree_file, "w");
            if (tree_filePtr) {
                save_tree_to_file(root, tree_filePtr);
                fclose(tree_filePtr);
            }

            fclose(arrayFile);
            free_tree(root); //Free memory used by current tree
            root = NULL; // Reseting root for next array
        }

        // Calculating & storing average rotations & height for current size
        char rotationFile[50], heightFile[50];
        sprintf(rotationFile, "avl_rotations_data_size%d.txt", array_size);
        sprintf(heightFile, "avl_height_data_size%d.txt", array_size);

        FILE *rotFile = fopen(rotationFile, "w");
        FILE *htFile = fopen(heightFile, "w");
        if (rotFile) {
            fprintf(rotFile, "Average Rotations: %f\n", rotations_total / 100);
            fclose(rotFile);
        }
        if (htFile) {
            fprintf(htFile, "Average Height: %f\n", height_total / 100);
            fclose(htFile);
        }
    }

    return 0;
}

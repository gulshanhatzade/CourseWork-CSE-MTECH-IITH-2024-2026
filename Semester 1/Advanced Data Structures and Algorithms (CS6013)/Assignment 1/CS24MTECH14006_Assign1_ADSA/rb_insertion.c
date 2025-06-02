//Red Black Tree Insertion
#include <stdio.h>
#include <stdlib.h>

typedef enum { RED, BLACK } Color;

typedef struct Node {
    int data;   // Data stored in  node
    Color color;    // Color of node either RED or BLACK
    struct Node *l, *r, *p; // Left child, right child & parentpointers
} Node;

Node *root = NULL;

// Function for creating new node with given data
Node* create_node(int data) {
    Node *new_node = (Node*)malloc(sizeof(Node)); //Allocating memory for new node
    new_node->data = data;
    new_node->color = RED; // New nodes are red default
    new_node->l = new_node->r = new_node->p = NULL;
    return new_node;
}

// Free memory of R B tree
void freeTree(Node *node) {
    if (node != NULL) {
        freeTree(node->l);  // Freeing left subtree
        freeTree(node->r);  // Freeing right subtree
        free(node);
    }
}

// Left rotation function for balancing R-B tree
void rotate_L(Node **root, Node *x, int *rotations) {
    Node *y = x->r; // Set y to be right child of x
    x->r = y->l;    // Move left subtree of y toright subtree of x

    if (y->l != NULL)
        y->l->p = x;    // Updating parent pointer of y's left child, if it exists

    y->p = x->p;    // Make parent of y same as parent of x

    if (x->p == NULL)
        *root = y; // Updating root
    else if (x == x->p->l)
        x->p->l = y;
    else
        x->p->r = y;

    y->l = x;
    x->p = y;

    (*rotations)++; // Incrementing count of rotation
}

// Right rotation function for balancing RB tree
void rotate_R(Node **root, Node *y, int *rotations) {
    Node *x = y->l; // Set x to be left child of y
    y->l = x->r;    // Moveright subtree of x to left subtree of y

    if (x->r != NULL)
        x->r->p = y;    // Updating parent pointer of x's right child, if it exists

    x->p = y->p;

    if (y->p == NULL)
        *root = x; // Updating root
    else if (y == y->p->r)
        y->p->r = x;
    else
        y->p->l = x;

    x->r = y;
    y->p = x;

    (*rotations)++; // Incrementing count of rotation
}

// Fix violations after r_b_insertion for ensuring RBtree properties are maintained
void fix_violation_rb(Node **root, Node *z, int *rotations) {
    Node *y;
    while (z != *root && z->p->color == RED) {  // Checking for violation means check parent is red
        if (z->p == z->p->p->l) {   // If parent is  left child
            y = z->p->p->r; // y is the uncle of z
            if (y != NULL && y->color == RED) {// Case 1- Uncle is red
                z->p->color = BLACK;
                y->color = BLACK;
                z->p->p->color = RED;
                z = z->p->p;
            } else {
                if (z == z->p->r) { // Case 2- z is a right child
                    z = z->p;
                    rotate_L(root, z, rotations);
                }
                z->p->color = BLACK;    // Case 3-z is a left child, recolor parent
                z->p->p->color = RED;
                rotate_R(root, z->p->p, rotations);
            }
        } else {    // If parent is a right child
            y = z->p->p->l;
            if (y != NULL && y->color == RED) { // Case 1- Uncle is red
                z->p->color = BLACK;
                y->color = BLACK;
                z->p->p->color = RED;
                z = z->p->p;
            } else {
                if (z == z->p->l) { // Case 2- z is a left child

                    z = z->p;
                    rotate_R(root, z, rotations);
                }
                z->p->color = BLACK;    // Case 3- z is a right child, recolor parent
                z->p->p->color = RED;
                rotate_L(root, z->p->p, rotations);
            }
        }
    }
    (*root)->color = BLACK; // Ensuring root is always black
}

// Inserting a node into the RB tree
void r_b_insert(Node **root, int data, int *rotations, int *height) {
    Node *z = create_node(data);    // Creating new node
    Node *y = NULL; 
    Node *x = *root;
    int track_local_height = 0; // For track height during insertion


    while (x != NULL) { // Traversing tree for finding right spot
        y = x;
        if (z->data < x->data) {
            x = x->l;
        } else {
            x = x->r;
        }
        track_local_height++;
    }

    z->p = y;   // Set parent of new node

    if (y == NULL) {
        *root = z; // Tree was empty
    } else if (z->data < y->data) {
        y->l = z;
    } else {
        y->r = z;
    }

    fix_violation_rb(root, z, rotations);   // Fix any R B tree violations


    if (track_local_height > *height) {
        *height = track_local_height; // Updating height
    }
}

// Saving R B tree to a file by pre order traversal
void save_tree_to_file(Node *root, FILE *file) {
    if (root != NULL) {
        fprintf(file, "%d %s\n", root->data, root->color == RED ? "RED" : "BLACK");
        save_tree_to_file(root->l, file);   // Recur on left subtree
        save_tree_to_file(root->r, file);   // Recur on right subtree
    }
}



int main() {
    int sizes[] = {10000, 100000, 1000000, 10000000}; // Sizes of the arrays
    
    for (int s = 0; s < 4; s++) {
        int array_size = sizes[s];
        int rotations_total = 0;
        int height_total = 0;

        // Perform operations for 100 arrays of current size
        for (int i = 1; i <= 100; i++) { 
            char file_name[50];
            sprintf(file_name, "array_size%d_%d.txt", array_size, i); // Create file_name
            FILE *arrayFile = fopen(file_name, "r");
            if (!arrayFile) {
                printf("Failed to open file: %s\n", file_name);
                continue;
            }

            int value;
            int rotations = 0;
            int height = 0;

            // Inserting each value from file into R B tree
            while (fscanf(arrayFile, "%d", &value) != EOF) {
                r_b_insert(&root, value, &rotations, &height);
            }

            // Updating total rotations & height for this size
            rotations_total += rotations;
            height_total += height;

            // Saving R B tree structure to a file
            char tree_file[50];
            sprintf(tree_file, "red_black_tree_size%d_%d.txt", array_size, i);
            FILE *tree_filePtr = fopen(tree_file, "w");
            if (tree_filePtr) {
                save_tree_to_file(root, tree_filePtr);   // Saving tree to file
                fclose(tree_filePtr);
            }

            fclose(arrayFile);
            freeTree(root); // Free memory of tree after each iteration
            root = NULL;    // Reset root for next array
        }

        // Save average rotations & height for this array size
        char averageFile[50];
        sprintf(averageFile, "average_rotations_height_size%d.txt", array_size);
        FILE *avgFile = fopen(averageFile, "w");
        if (avgFile) {
            fprintf(avgFile, "Average Rotations: %f\n", (float)rotations_total / 100);
            fprintf(avgFile, "Average Height: %f\n", (float)height_total / 100);
            fclose(avgFile);
        }
    }

    return 0;
}

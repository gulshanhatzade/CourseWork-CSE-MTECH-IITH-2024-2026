#include <stdio.h>
#include <stdlib.h>

//Structure of avl tree nodes
typedef struct Node {
    int data;   //value of node
    struct Node *l,*r;  //Pointers to child
    int height; //height of node
} Node;

Node *root = NULL; //Global variable - root of avl tree



// Create a new node
Node* create_node(int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    newNode->l = NULL;
    newNode->r = NULL;
    newNode->height = 1; // New nodes are initially added at leaf level
    return newNode;
}
int maximum(int a, int b);
int height(Node *N);

// Function for performing r rotation around node y
Node* r_Rotate(Node *y, int *rotations) {
    Node *x = y->l;
    Node *T2 = x->r;

    // Perform r rotation
    x->r = y;
    y->l = T2;

    // Updating heights of y and x after rotation
    y->height = maximum(height(y->l), height(y->r)) + 1;
    x->height = maximum(height(x->l), height(x->r)) + 1;

    (*rotations)++; // Increment rotation count for tracking

    // Returning the new root
    return x;
}

// Function for performing lrotation around node x
Node* l_Rotate(Node *x, int *rotations) {
    Node *y = x->r;
    Node *T2 = y->l;

    // Perform lrotation
    y->l = x;
    x->r = T2;

    // Updating heights of x and y after rotation
    x->height = maximum(height(x->l), height(x->r)) + 1;
    y->height = maximum(height(y->l), height(y->r)) + 1;

    (*rotations)++; // Increment rotation count for tracking

    // Returning the new root
    return y;
}

//Function for getting maximum of two integers
int maximum(int a, int b) {
    return (a > b) ? a : b;
}


// Function for getting height of node in avl tree
int height(Node *N) {
    if (N == NULL)
        return 0;
    return N->height;
}

// Function for getting balance factor of a node
int get_balance(Node *N) {
    if (N == NULL)
        return 0;
    return height(N->l) - height(N->r); //Diff b/n heights of land r subtrees
}





// Function for inseting new value in the AVL tree and rebalance
Node* insert_avl(Node* node, int data, int *rotations) {
    // Step 1- Perform normal B S T insertion
    if (node == NULL)
        return create_node(data);

    if (data < node->data)
        node->l = insert_avl(node->l, data, rotations);
    else if (data > node->data)
        node->r = insert_avl(node->r, data, rotations);
    else
        return node; // Duplicates are not allowed in AVL tree

   // Step 2- Updating height of current node
    node->height = 1 + maximum(height(node->l), height(node->r));

    // Step 3- Getting balance factor to check if the node is unbalanced
    int balance = get_balance(node);

    // Step 4- Rebalancing node if necessary by performing rotations
    if (balance > 1 && data < node->l->data)
        return r_Rotate(node, rotations);

    if (balance < -1 && data > node->r->data)
        return l_Rotate(node, rotations);

    if (balance > 1 && data > node->l->data) {
        node->l = l_Rotate(node->l, rotations);
        return r_Rotate(node, rotations);
    }

    if (balance < -1 && data < node->r->data) {
        node->r = r_Rotate(node->r, rotations);
        return l_Rotate(node, rotations);
    }

    return node;
}




// Function for finding node with minimum value
Node* min_value_node(Node* node) {
    Node* current_node = node;
    while (current_node && current_node->l != NULL) {
        current_node = current_node->l;
    }
    return current_node;
}

// Function to delete a node
Node* delete_node(Node* root, int data, int *rotations) {
    if (root == NULL)
        return root;

    // Step 1- Perform standard B S T deletion
    if (data < root->data)
        root->l= delete_node(root->l, data, rotations);
    else if (data > root->data)
        root->r = delete_node(root->r, data, rotations);
    else {
        // Case 1- Node with only 1 child or no child
        if ((root->l== NULL) || (root->r == NULL)) {
            Node* temp = root->l? root->l: root->r; // Use non null child if present

            // No child case
            if (temp == NULL) {
                // Case 1a- No children, the node is a leaf
                temp = root;
                root = NULL; // This will make parent point to NULL
            } else {
                // Case 1b- One child, copy the child's data to current node
                *root = *temp; // Copy the contents of non empty child
            }
            free(temp); // Freeing original node
        } else {
            // Case 2- Node with 2 children
            //  Getting inorder successor (smallest in r subtree)
            Node* temp = min_value_node(root->r);
            root->data = temp->data; // Copy inorder successor's data to this node
            root->r = delete_node(root->r, temp->data, rotations); // Delete  inorder successor
        }
    }

    // If  tree had only one node, then return
    if (root == NULL)
        return root;

    // Update height of current node
    root->height = 1 + maximum(height(root->l), height(root->r));

    // Get balancing factor
    int balance = get_balance(root);

    // If node becomes unbalanced, then there are 4 cases

    if (balance > 1 && get_balance(root->l) >= 0)
        return r_Rotate(root, rotations);

    if (balance > 1 && get_balance(root->l) < 0) {
        root->l= l_Rotate(root->l, rotations);
        return r_Rotate(root, rotations);
    }

    if (balance < -1 && get_balance(root->r) <= 0)
        return l_Rotate(root, rotations);

    if (balance < -1 && get_balance(root->r) > 0) {
        root->r = r_Rotate(root->r, rotations);
        return l_Rotate(root, rotations);
    }

    return root;
}

// Free memory of the AVL tree
void free_tree(Node *node) {
    if (node != NULL) {
        free_tree(node->l);
        free_tree(node->r);
        free(node);
    }
}

int main() {
    int sizes[] = {10000, 100000, 1000000, 10000000}; // Sizes of the arrays

    for (int s = 0; s < 4; s++) {
        int array_size = sizes[s];
        float rotations_total = 0;
        float height_total = 0;

        // Process 100 different arrays, for each size
        for (int i = 1; i <= 100; i++) { 
            char filename[50];
            sprintf(filename, "avl_tree_size%d_%d.txt", array_size, i); // Use pre-generated AVL tree
            FILE *arrayFile = fopen(filename, "r");
            if (!arrayFile) {
                printf("Failed to open file- %s\n", filename);
                continue;
            }

            int rotations = 0;
            int value;

            // Inserting AVL nodes into the AVL tree
            while (fscanf(arrayFile, "%d", &value) != EOF) {
                root = insert_avl(root, value, &rotations);
            }

            fclose(arrayFile);

            // Deleting about 1/10th of elements randomly
            int elements_to_delete = array_size / 10;
            for (int j = 0; j < elements_to_delete; j++) {
                // Generating a random value to delete
                int random_value = rand() % (array_size * 2);
                root = delete_node(root, random_value, &rotations);
            }

            rotations_total += rotations;   // Accumulate rotations for current array
            height_total += height(root);   //Accumulate final height of the tree


            free_tree(root); //Free memory used by current tree
            root = NULL; // Reseting root
        }

        // Calculating & storing average rotations & height for current size
        char deletion_file[50];
        sprintf(deletion_file, "avl_deletion_data_size%d.txt", array_size);
        FILE *delFile = fopen(deletion_file, "w");
        if (delFile) {
            fprintf(delFile, "Average Rotations: %f\n", rotations_total / 100);
            fprintf(delFile, "Average Height: %f\n", height_total / 100);
            fclose(delFile);
        }
    }

    return 0;
}

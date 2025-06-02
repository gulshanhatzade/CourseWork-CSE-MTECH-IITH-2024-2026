// Problem 0
#include <stdio.h>
#include <time.h> //For seeding random no. generator
#include <stdlib.h> // for random function, memory allocation

// Function for generating random permutation of 1 to n
void random_array_generate(int size, int a[]) {

    // Fill the array with numbers from 1 to n
    for (int i = 0; i < size; i++) {
        a[i] = i + 1;
    }

    //Shuffling the array
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1); // Generates random number & ensures random no. is in the range [0,i]
        
        //Swaping a[i] & a[j]
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

// Function for saving array to file
void array_save_to_file(const char* filename, int a[], int size) {
    FILE* file = fopen(filename, "w"); //Opens a file with write mode & returns file pointer or creates it if file doesnt exist.
    if (file == NULL) {
        printf("Failed to open the file for writing.\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d\n", a[i]); // Write each element of array to file
    }
    fclose(file);
}

int main() {
    srand(time(0));  // Seed random number generator with current time

    int diff_sizes[] = {10000, 100000, 1000000, 10000000}; //Array of different sizes

    for (int s = 0; s < 4; s++) // loop iterates over each size
    {
        int size = diff_sizes[s]; //fetches the current size
        printf("Generating 100 random arrays of size %d & saving to files\n", size);

        for (int i = 0; i < 100; i++) //For generating 100 random arrays for current size
        {
            int* arr = (int*)malloc(size * sizeof(int));  // Allocate memory dynamically for the array
            random_array_generate(size, arr);  // Fill the array with random permution from 1 to size


            // Saving array to a file
            char filename[50];

            //Creating unique filename for each array using
            sprintf(filename, "array_size%d_%d.txt", size, i + 1);

            //Calling funtion for writing content of array to file with generated filename
            array_save_to_file(filename, arr, size);

            free(arr);  // Freeing allocated memory for array
        }
    }

    return 0;
}

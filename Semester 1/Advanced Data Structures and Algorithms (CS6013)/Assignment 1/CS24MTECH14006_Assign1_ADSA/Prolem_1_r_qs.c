//Problem 1 Randomized Quick Sort
#include <stdio.h>
#include <time.h>       //For seeding random no. generator
#include <math.h>       // Provides mathematical functions like sqrt for calculating square root
#include <stdlib.h>     // for random function, memory allocation

int comparisons = 0;    //Global variable to keep track of comparisons

// swap function
void swap_integers(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

// Partition function for sorting using quick sort
int partition(int a[], int low, int high) {
    int pivot = a[high];  // Choosing last element as pivot
    int i = low - 1;
    for (int j = low; j < high; j++) {
        comparisons++;
        if (a[j] <= pivot) {
            i++;
            swap_integers(&a[i], &a[j]);    // Swaping current element with smaller one
        }
    }
    swap_integers(&a[i + 1], &a[high]); // Swaping pivot into its correct position
    return (i + 1);     // Returning index of pivot
}

// Randomized partition function
int random_partition(int a[], int low, int high) {
    int random_index = low + rand() % (high - low);   // Picking an index randomly
    swap_integers(&a[random_index], &a[high]);    // Swaping randomly picked element with pivot
    return partition(a, low, high);           // Calling standard partition function
}

// Randomized quicksort function
void randomized_quicksort(int a[], int low, int high) {
    if (low < high) {
        int partition_index = random_partition(a, low, high);    // Getting partition index
        randomized_quicksort(a, low, partition_index - 1);  // Recursively sorting left subarray
        randomized_quicksort(a, partition_index + 1, high); // Recursively sorting right subarray
    }
}

int main() {
    int sizes[] = {10000, 100000, 1000000, 10000000};
    char filename[50];          //For storing file name of the input array
    char comparison_file[50];   //For storing file name for saving comparisons
    srand(time(0));

    // Open file to save the average comparisons & standard deviation
    FILE *average_output = fopen("average_comparisons.txt", "w");
    if (average_output == NULL) {
        printf("Failed to open file to save average comparisons.\n");
        return 1;
    }

    for (int i = 0; i < 4; i++) {
        int size = sizes[i];
        double comparisons_total = 0;   // For storing  total comparisons over 100 trials
        double square_comparisons = 0;     // Fors storing square of comparisons for calculation of  variance 

        // Open file for saving comparisons for this array size
        sprintf(comparison_file, "comparisons_size%d.txt", size);   // Creating filename for comparison results
        FILE *output_comparison = fopen(comparison_file, "w");

        if (output_comparison == NULL) {
            printf("Failed to open file to save comparisons for size %d\n", size);
            return 1;
        }


        //Inner Loop for Each Trial
        for (int j = 1; j <= 100; j++) {
            sprintf(filename, "array_size%d_%d.txt", size, j);   // Creating filename for input array
            FILE *file = fopen(filename, "r");
            if (file == NULL) {
                printf("Failed to open array file- %s\n", filename);
                return 1;
            }

            int *arr = (int *)malloc(size * sizeof(int));   // Allocating memory for array
            if (arr == NULL) {
                printf("For size %d memory allocation is failed.\n", size);
                return 1;
            }

            for (int k = 0; k < size; k++)
                fscanf(file, "%d", &arr[k]);    // Reading array elements from file

            fclose(file);

            comparisons = 0;        // Reseting comparison counter
            randomized_quicksort(arr, 0, size - 1);     // Sorting array using randomized quicksort
            comparisons_total += comparisons;
            square_comparisons += (double)comparisons * comparisons;

            // Saving no. of comparisons for this trial to file
            fprintf(output_comparison, "%d\n", comparisons);

            // Freeing memory after each trial
            free(arr);

            // Printing progress for each trial
            printf("Processed size %d with trial number %d\n", size, j);
        }

        double mean = comparisons_total / 100;      // Calculating average comparisons
        double variance = (square_comparisons / 100) - (mean * mean);

        // Ensuring non-negative variance
        if (variance < 0) {
            variance = 0;
        }

        double standard_deviation = sqrt(variance);

        // Printing the  results
        printf("Array size- %d, Average comparisons- %.2f, Standard deviation- %.2f\n", size, mean, standard_deviation);

        // Save average comparisons and standard deviation to average_output file
        fprintf(average_output, "Array size: %d, Avg Comparisons: %.2f, Std Dev: %.2f\n", size, mean, standard_deviation);

        // Close file after all trials for this size
        fclose(output_comparison);
    }

    // Close average_output file after writing all results
    fclose(average_output);

    return 0;
}

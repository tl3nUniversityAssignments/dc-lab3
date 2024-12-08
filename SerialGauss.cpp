#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int* pSerialPivotPos;  // Number of pivot rows selected at the iterations
int* pSerialPivotIter; // Iterations, at which the rows were pivots

// Function for simple initialization of the matrix and the vector elements
void DummyDataInitialization(double* pMatrix, double* pVector, int Size) {
    int i, j; // Loop variables
    for (i = 0; i < Size; i++) {
        pVector[i] = i + 1;
        for (j = 0; j < Size; j++) {
            if (j <= i)
                pMatrix[i * Size + j] = 1;
            else
                pMatrix[i * Size + j] = 0;
        }
    }
}

// Function for random initialization of the matrix and the vector elements
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
    int i, j; // Loop variables
    srand(unsigned(clock()));
    for (i = 0; i < Size; i++) {
        pVector[i] = rand() / double(1000);
        for (j = 0; j < Size; j++) {
            if (j <= i)
                pMatrix[i * Size + j] = rand() / double(1000);
            else
                pMatrix[i * Size + j] = 0;
        }
    }
}

// Function for memory allocation and data initialization
void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, int& Size) {
    // Setting the size of the matrix and the vector
    while (Size <= 0) {
        printf("\nEnter the size of the matrix and the vector: ");
        scanf("%d", &Size);
        printf("\nChosen size = %d \n", Size);
        if (Size <= 0)
            printf("\nSize of objects must be greater than 0!\n");
    }

    // Memory allocation
    pMatrix = new double[Size * Size];
    pVector = new double[Size];
    pResult = new double[Size];

    // Initialization of the matrix and the vector elements
    // DummyDataInitialization(pMatrix, pVector, Size);
    RandomDataInitialization(pMatrix, pVector, Size);
}

// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++) {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * RowCount + j]);
        printf("\n");
    }
}

// Function for formatted vector output
void PrintVector(double* pVector, int Size) {
    int i;
    for (i = 0; i < Size; i++)
        printf("%7.4f ", pVector[i]);
}

// Function for finding the pivot row
int FindPivotRow(double* pMatrix, int Size, int Iter) {
    int PivotRow = -1; // Index of the pivot row
    int MaxValue = 0;  // Value of the pivot element
    int i;             // Loop variable
    // Choose the row that stores the maximum element
    for (i = 0; i < Size; i++) {
        if ((pSerialPivotIter[i] == -1) && (fabs(pMatrix[i * Size + Iter]) > MaxValue)) {
            PivotRow = i;
            MaxValue = fabs(pMatrix[i * Size + Iter]);
        }
    }
    return PivotRow;
}

// Function for the column elimination
void SerialColumnElimination(double* pMatrix, double* pVector, int Pivot, int Iter, int Size) {
    double PivotValue, PivotFactor;
    PivotValue = pMatrix[Pivot * Size + Iter];
    for (int i = 0; i < Size; i++) {
        if (pSerialPivotIter[i] == -1) {
            PivotFactor = pMatrix[i * Size + Iter] / PivotValue;
            for (int j = Iter; j < Size; j++) {
                pMatrix[i * Size + j] -= PivotFactor * pMatrix[Pivot * Size + j];
            }
            pVector[i] -= PivotFactor * pVector[Pivot];
        }
    }
}

// Function for the Gaussian elimination
void SerialGaussianElimination(double* pMatrix, double* pVector, int Size) {
    int Iter;     // Number of the iteration of the Gaussian elimination
    int PivotRow; // Number of the current pivot row
    for (Iter = 0; Iter < Size; Iter++) {
        // Finding the pivot row
        PivotRow = FindPivotRow(pMatrix, Size, Iter);
        pSerialPivotPos[Iter] = PivotRow;
        pSerialPivotIter[PivotRow] = Iter;
        SerialColumnElimination(pMatrix, pVector, PivotRow, Iter, Size);
    }
}

// Function for the back substitution
void SerialBackSubstitution(double* pMatrix, double* pVector, double* pResult, int Size) {
    int RowIndex, Row;
    for (int i = Size - 1; i >= 0; i--) {
        RowIndex = pSerialPivotPos[i];
        pResult[i] = pVector[RowIndex] / pMatrix[Size * RowIndex + i];
        for (int j = 0; j < i; j++) {
            Row = pSerialPivotPos[j];
            pVector[j] -= pMatrix[Row * Size + i] * pResult[i];
            pMatrix[Row * Size + i] = 0;
        }
    }
}

// Function for the execution of the Gauss algorithm
void SerialResultCalculation(double* pMatrix, double* pVector, double* pResult, int Size) {
    // Memory allocation
    pSerialPivotPos = new int[Size];
    pSerialPivotIter = new int[Size];
    for (int i = 0; i < Size; i++) {
        pSerialPivotIter[i] = -1;
    }
    // Gaussian elimination
    SerialGaussianElimination(pMatrix, pVector, Size);
    // Back substitution
    SerialBackSubstitution(pMatrix, pVector, pResult, Size);
    // Memory deallocation
    delete[] pSerialPivotPos;
    delete[] pSerialPivotIter;
}

// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pVector, double* pResult) {
    delete[] pMatrix;
    delete[] pVector;
    delete[] pResult;
}

void test() {
    double* pMatrix; // Matrix of the linear system
    double* pVector; // Right parts of the linear system
    double* pResult; // Result vector
    int sizes[] = {10, 100, 500, 1000, 1500, 2000, 2500, 3000};
    time_t start, finish;
    double duration;
    printf("Serial Gauss algorithm for solving linear systems\n");
    for (int i = 0; i < 8; i++) {
        int Size = sizes[i];
        // Memory allocation and definition of objects' elements
        ProcessInitialization(pMatrix, pVector, pResult, Size);
        printf("Size = %d\n", Size);
        // The matrix and the vector output
        //printf("Initial Matrix \n");
        //PrintMatrix(pMatrix, Size, Size);
        //printf("Initial Vector \n");
        //rintVector(pVector, Size);
        // Execution of the Gauss algorithm
        start = clock();
        SerialResultCalculation(pMatrix, pVector, pResult, Size);
        finish = clock();
        duration = (finish - start) / double(CLOCKS_PER_SEC);
        // Printing the result vector
        //printf("\n Result Vector: \n");
        //PrintVector(pResult, Size);
        // Printing the execution time of the Gauss method
        printf("\n Time of execution: %f\n", duration);
        // Computational process termination
        ProcessTermination(pMatrix, pVector, pResult);
    }
}

int main() {
    // double* pMatrix; // Matrix of the linear system
    // double* pVector; // Right parts of the linear system
    // double* pResult; // Result vector
    // int Size;        // Sizes of the initial matrix and the vector
    // time_t start, finish;
    // double duration;
    // printf("Serial Gauss algorithm for solving linear systems\n");
    // // Memory allocation and definition of objects' elements
    // ProcessInitialization(pMatrix, pVector, pResult, Size);
    // // The matrix and the vector output
    // printf("Initial Matrix \n");
    // PrintMatrix(pMatrix, Size, Size);
    // printf("Initial Vector \n");
    // PrintVector(pVector, Size);
    // // Execution of the Gauss algorithm
    // start = clock();
    // SerialResultCalculation(pMatrix, pVector, pResult, Size);
    // finish = clock();
    // duration = (finish - start) / double(CLOCKS_PER_SEC);
    // // Printing the result vector
    // printf("\n Result Vector: \n");
    // PrintVector(pResult, Size);
    // // Printing the execution time of the Gauss method
    // printf("\n Time of execution: %f\n", duration);
    // // Computational process termination
    // ProcessTermination(pMatrix, pVector, pResult);

    test();
    return 0;
}
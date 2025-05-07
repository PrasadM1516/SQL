#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void printArray(const vector<int>& arr) {
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
}

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp;
    int i = left, j = mid + 1;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j])
            temp.push_back(arr[i++]);
        else
            temp.push_back(arr[j++]);
    }

    while (i <= mid)
    {
        temp.push_back(arr[i++]);
    } 
    while (j <= right)
    {
        temp.push_back(arr[j++]);
    } 

    for (int k = 0; k < temp.size(); ++k)
    {
        arr[left + k] = temp[k];
    }
}

void mergeSortSequential(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void mergeSortParallel(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr, left, mid);
            #pragma omp section
            mergeSortParallel(arr, mid + 1, right);
        }
        merge(arr, left, mid, right);
    }
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    vector<int> arr(n);
    cout << "Enter elements: ";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    vector<int> arrSeq = arr;
    vector<int> arrPar = arr;
    double start,end;

    // Time for Sequential Merge Sort
    start = omp_get_wtime();
    mergeSortSequential(arrSeq, 0, n - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << (end - start) * 1000 << " milliseconds\n";
    cout << "Sorted Array (Merge Sort - Sequential): ";
    printArray(arrSeq);

    // Time for Parallel Merge Sort
    start = omp_get_wtime();
    mergeSortParallel(arrPar, 0, n - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << (end - start) * 1000 << " milliseconds\n";
    cout << "Sorted Array (Merge Sort - Parallel): ";
    printArray(arrPar);

    return 0;
}

#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void printArray(const vector<int>& arr)
{
    for(int num: arr)
    {
        cout<< num <<" ";
    }
    cout<<endl;
}

void bubbleSortSequential(vector<int>& arr)
{
    int n = arr.size();
    for(int i=0; i< n-1; i++)
    {
        for(int j=0; j < n-i-1; j++)
        {
            if(arr[j] > arr[j+1])
            {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

void bubbleSortParallel(vector<int>& arr)
{
    int n = arr.size();
    #pragma omp parallel for
    for(int i=0; i< n-1; i++)
    {
        for(int j=0; j < n-i-1; j++)
        {
            if(arr[j] > arr[j+1])
            {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}


int main()
{
    int n;
    cout<<"Enter the number of elements: ";
    cin>>n;

    vector<int> arr(n);
    cout<<"Enter elements: ";
    for(int i=0; i<n; i++)
    {
        cin>>arr[i];
    }

    vector<int> arrSeq = arr;
    vector<int> arrPar = arr;
    double start,end;

    start = omp_get_wtime();
    bubbleSortSequential(arrSeq);
    end = omp_get_wtime();
    cout<<"Sequential Sorted Array : ";
    printArray(arrSeq);
    cout<<"Sequential Bubble Sort Time : "<< end-start<<" seconds.\n";

    start = omp_get_wtime();
    bubbleSortParallel(arrPar);
    end = omp_get_wtime();
    cout<<"Parallel Sorted Array : ";
    printArray(arrPar);
    cout<<"Parallel Bubble Sort Time : "<< end-start<<" seconds.";

    return 0;
}


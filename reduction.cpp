#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
using namespace std;

void min_reduction(vector<int>& arr)
{
    double start = omp_get_wtime();
    int min_value = INT_MAX;
    #pragma omp parallel for reduction(min: min_value)
    for(int i=0; i< arr.size(); i++)
    {
        if(arr[i] < min_value)
        {
            min_value = arr[i];
        }
    }
    double end = omp_get_wtime();
    cout << "Minimum value: " << min_value << " (Time taken: " << end - start << " seconds)" << endl;
}

void max_reduction(vector<int>& arr)
{
    double start = omp_get_wtime();
    int max_value = INT_MIN;
    #pragma omp parallel for reduction(max: max_value)
    for(int i=0; i< arr.size(); i++)
    {
        if(arr[i] > max_value)
        {
            max_value = arr[i];
        }
    }
    double end = omp_get_wtime();
    cout << "Maximum value: " << max_value << " (Time taken: " << end - start << " seconds)" << endl;
}

void sum_reduction(vector<int>& arr)
{
    double start = omp_get_wtime();
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for(int i=0; i< arr.size(); i++)
    {
        sum += arr[i];
    }
    double end = omp_get_wtime();
    cout << "Sum: " << sum << " (Time taken: " << end - start << " seconds)" << endl;
}

void avg_reduction(vector<int>& arr)
{
    double start = omp_get_wtime();
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for(int i=0; i< arr.size(); i++)
    {
        sum += arr[i];
    }
    double end = omp_get_wtime();
    cout << "Average: " << (double)sum/arr.size() << " (Time taken: " << end - start << " seconds)" << endl;
}

int main()
{
    int n;
    cout<<"Enter number of elements: ";
    cin>>n;

    vector<int> arr(n);
    cout<<"Enter elements: ";
    for(int i=0; i<n; i++)
    {
        cin>>arr[i];
    }

    min_reduction(arr);
    max_reduction(arr);
    sum_reduction(arr);
    avg_reduction(arr);

    return 0;
}


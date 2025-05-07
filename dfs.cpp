#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

class ParallelDFS {
private:
    vector<vector<int>> adjMatrix;
    vector<int> visited;
    int n;

public:
    void input() {
        cout << "Enter the number of vertices: ";
        cin >> n;
        adjMatrix.resize(n, vector<int>(n));
        visited.resize(n);
        cout << "Enter the adjacency matrix:\n";
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                cin >> adjMatrix[i][j];
    }

    void dfs_sequential(int start) {
        stack<int> s;
        fill(visited.begin(), visited.end(), 0);  // Reset visited array
        s.push(start);
        visited[start] = 1;
        cout << "Sequential DFS Order: ";

        while (!s.empty()) {
            int node = s.top();
            s.pop();
            cout << node << " ";

            // Explore neighbors of the current node
            for (int i = 0; i < n; i++) {
                if (adjMatrix[node][i] && !visited[i]) {
                    s.push(i);
                    visited[i] = 1;
                }    
            }
        }
        cout << endl;
    }

    void dfs_parallel(int start) {
        stack<int> s;
        fill(visited.begin(), visited.end(), 0);  // Reset visited array
        visited[start] = 1;
        cout << "Parallel DFS Order: ";
        
        s.push(start);

        // Start DFS
        while (!s.empty()) {
            int node = s.top();
            s.pop();
            // Print the node immediately as it is visited
            cout << node << " ";
            // Parallelize the exploration of neighbors
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                if (adjMatrix[node][i] && !visited[i]) {
                // If the neighbor is unvisited and there is an edge, push it to the stack
                    #pragma omp critical
                    {
                        if (!visited[i]) {
                            s.push(i);
                            visited[i] = 1;
                        }
                    }
                }
            }    
        }
        cout << endl;
    }
};

int main() {
    ParallelDFS dfs;
    dfs.input();
    int startVertex;
    cout << "Enter the starting vertex for DFS: ";
    cin >> startVertex;

    double start, end;

    // Sequential DFS
    start = omp_get_wtime();
    dfs.dfs_sequential(startVertex);
    end = omp_get_wtime();
    cout << "Time taken by Sequential DFS: " << end - start << " seconds\n";

    // Parallel DFS
    start = omp_get_wtime();
    dfs.dfs_parallel(startVertex);
    end = omp_get_wtime();
    cout << "Time taken by Parallel DFS: " << end - start << " seconds\n";

    return 0;
}

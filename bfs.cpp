#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

class ParallelBFS {
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

    void bfs_sequential(int start) {
        queue<int> q;
        fill(visited.begin(), visited.end(), 0);
        q.push(start);
        visited[start] = 1;
        cout << "Sequential BFS Order: ";
        while (!q.empty()) {
            int node = q.front(); 
            q.pop();
            cout << node << " ";
            for (int i = 0; i < n; i++) {
                if (adjMatrix[node][i] && !visited[i]) {
                    q.push(i);
                    visited[i] = 1;
                }
            }
        }
        cout << endl;
    }

    void bfs_parallel(int start) {
        queue<int> q;
        fill(visited.begin(), visited.end(), 0);
        q.push(start);
        visited[start] = 1;
        cout << "Parallel BFS Order: ";
        while (!q.empty()) {
            int node = q.front(); 
            q.pop();
            cout << node << " ";
            #pragma omp parallel for
                for (int i = 0; i < n; i++) {
                    if (adjMatrix[node][i] && !visited[i]) {
                        q.push(i);
                        visited[i] = 1;
                    }
                }   
        }
        cout << endl;
    }
};

int main() {
    ParallelBFS bfs;
    bfs.input();
    int startVertex;
    cout << "Enter the starting vertex for BFS: ";
    cin >> startVertex;

    double start, end;

    start = omp_get_wtime();
    bfs.bfs_sequential(startVertex);
    end = omp_get_wtime();
    cout << "Time taken by Sequential BFS: " << end - start << " seconds\n";

    start = omp_get_wtime();
    bfs.bfs_parallel(startVertex);
    end = omp_get_wtime();
    cout << "Time taken by Parallel BFS: " << end - start << " seconds\n";

    return 0;
}

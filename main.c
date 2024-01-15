#include <stdio.h>
#include <omp.h>

#define V 5

int out_degree(int vertex, int graph[V][V]) {
    int count = 0;
    for (int i = 0; i < V; ++i) {
        if (graph[vertex][i] != 0) {
            count++;
        }
    }
    return count;
}

void mpbc_algorithm(int graph[V][V], int vertices) {
    double betweenness_centrality[V];

    #pragma omp parallel for
    for (int source = 0; source < vertices; ++source) {
        double messages[V][V] = {{0.0}};
        double dependency[V] = {0.0};

        // Message passing
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < vertices; ++i) {
            for (int j = 0; j < vertices; ++j) {
                if (graph[i][j] != 0) {
                    double msg = messages[source][i] * (1.0 / out_degree(i, graph));
                    #pragma omp atomic
                    messages[source][j] += msg;
                }
            }
        }

        // Dependency accumulation
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < vertices; ++i) {
            for (int j = 0; j < vertices; ++j) {
                if (graph[i][j] != 0) {
                    double dep = messages[source][i] * (1.0 / out_degree(i, graph));
                    #pragma omp atomic
                    dependency[j] += dep;
                }
            }
        }

        // Accumulate betweenness centrality
        #pragma omp parallel for
        for (int i = 0; i < vertices; ++i) {
            #pragma omp atomic
            betweenness_centrality[i] += dependency[i];
        }
    }

    // Print Betweenness Centrality values
    printf("Betweenness Centrality:\n");
    for (int i = 0; i < vertices; ++i) {
        printf("%d: %f\n", i, betweenness_centrality[i]);
    }
}

int main() {
    // Sample graph initialization
    int graph[V][V] = {
        {0, 1, 1, 0, 0},
        {1, 0, 1, 1, 1},
        {1, 1, 0, 1, 1},
        {0, 1, 1, 0, 1},
        {0, 1, 1, 1, 0}
    };

    // Calling the MPBC algorithm
    mpbc_algorithm(graph, V);

    return 0;
}

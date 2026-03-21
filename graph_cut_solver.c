/*
 * graph_cut_solver.c
 * Dinic's max-flow/min-cut for grid graphs.
 * Called from Python via ctypes.
 *
 * Build:
 *   gcc -O3 -shared -fPIC -o graph_cut_solver.so graph_cut_solver.c
 */

#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------
 * Adjacency list using dynamic arrays
 * -------------------------------------------------------- */
typedef struct {
    int    to;   /* destination node */
    int    rev;  /* index of reverse edge in G[to]  */
    double cap;  /* residual capacity */
} Edge;

typedef struct {
    Edge *d;
    int   sz, cap;
} Vec;

/* --------------------------------------------------------
 * Graph state (module-level, reset on each call)
 * -------------------------------------------------------- */
static Vec *G     = NULL;
static int *lvl   = NULL;
static int *cur   = NULL;
static int  N_g   = 0;

static void graph_free(void) {
    if (!G) return;
    for (int i = 0; i < N_g; i++) if (G[i].d) free(G[i].d);
    free(G); free(lvl); free(cur);
    G = NULL; lvl = NULL; cur = NULL;
}

static void graph_init(int n) {
    graph_free();
    N_g = n;
    G   = (Vec *)calloc(n, sizeof(Vec));
    lvl = (int *)malloc(n * sizeof(int));
    cur = (int *)malloc(n * sizeof(int));
}

/* Push one edge-struct into a Vec */
static void vec_push(Vec *v, Edge e) {
    if (v->sz >= v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->d   = (Edge *)realloc(v->d, v->cap * sizeof(Edge));
    }
    v->d[v->sz++] = e;
}

/*
 * Add an edge pair:
 *   forward  u -> v  with capacity fwd_cap
 *   backward v -> u  with capacity rev_cap
 *
 * For symmetric (undirected) grid edges: fwd_cap == rev_cap
 * For directed terminal edges:           rev_cap == 0
 */
static void add_edge_pair(int u, int v, double fwd_cap, double rev_cap) {
    Edge ef = {v, G[v].sz, fwd_cap};
    Edge er = {u, G[u].sz, rev_cap};
    vec_push(&G[u], ef);
    vec_push(&G[v], er);
}

/* --------------------------------------------------------
 * Dinic's BFS — build level graph
 * -------------------------------------------------------- */
static int bfs(int s, int t) {
    memset(lvl, -1, N_g * sizeof(int));
    int *q    = (int *)malloc(N_g * sizeof(int));
    int  head = 0, tail = 0;

    lvl[s] = 0;
    q[tail++] = s;

    while (head < tail) {
        int v = q[head++];
        for (int i = 0; i < G[v].sz; i++) {
            Edge *e = &G[v].d[i];
            if (e->cap > 1e-9 && lvl[e->to] < 0) {
                lvl[e->to] = lvl[v] + 1;
                q[tail++]  = e->to;
            }
        }
    }
    free(q);
    return lvl[t] >= 0;
}

/* --------------------------------------------------------
 * Dinic's DFS with current-arc optimisation
 * -------------------------------------------------------- */
static double dfs(int v, int t, double f) {
    if (v == t) return f;
    for (; cur[v] < G[v].sz; cur[v]++) {
        Edge *e = &G[v].d[cur[v]];
        if (e->cap > 1e-9 && lvl[e->to] == lvl[v] + 1) {
            double d      = e->cap < f ? e->cap : f;
            double pushed = dfs(e->to, t, d);
            if (pushed > 1e-9) {
                e->cap                   -= pushed;
                G[e->to].d[e->rev].cap   += pushed;
                return pushed;
            }
        }
    }
    return 0.0;
}

/* --------------------------------------------------------
 * Public entry point — called from Python via ctypes
 *
 * Parameters
 * ----------
 * num_nodes  : total number of nodes (pixels + 2 for source/sink)
 * source     : index of the source node
 * sink       : index of the sink node
 * from_arr   : [num_edges] array of edge tail node indices
 * to_arr     : [num_edges] array of edge head node indices
 * fwd_cap    : [num_edges] forward capacities
 * rev_cap    : [num_edges] reverse capacities (0 for directed, same for symmetric)
 * num_edges  : number of edge pairs
 * segs_out   : [num_nodes] output — 1 if node is in the source-side segment
 *
 * Returns
 * -------
 * max-flow value (= min-cut cost)
 * -------------------------------------------------------- */
double solve_maxflow(
    int     num_nodes,
    int     source,
    int     sink,
    int    *from_arr,
    int    *to_arr,
    double *fwd_cap,
    double *rev_cap,
    int     num_edges,
    int    *segs_out
) {
    graph_init(num_nodes);

    for (int i = 0; i < num_edges; i++)
        add_edge_pair(from_arr[i], to_arr[i], fwd_cap[i], rev_cap[i]);

    /* Run Dinic's algorithm */
    double flow = 0.0;
    while (bfs(source, sink)) {
        memset(cur, 0, N_g * sizeof(int));
        double pushed;
        while ((pushed = dfs(source, sink, 1e18)) > 1e-9)
            flow += pushed;
    }

    /* BFS on residual to label source-side component */
    int *q    = (int *)malloc(N_g * sizeof(int));
    int  head = 0, tail = 0;
    memset(segs_out, 0, N_g * sizeof(int));
    segs_out[source] = 1;
    q[tail++]        = source;

    while (head < tail) {
        int v = q[head++];
        for (int i = 0; i < G[v].sz; i++) {
            Edge *e = &G[v].d[i];
            if (e->cap > 1e-9 && !segs_out[e->to]) {
                segs_out[e->to] = 1;
                q[tail++]       = e->to;
            }
        }
    }
    free(q);

    graph_free();
    return flow;
}

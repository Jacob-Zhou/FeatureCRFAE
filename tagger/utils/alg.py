import torch


def kmeans(x, k, max_it=32):
    """
    KMeans algorithm for clustering the sentences by length.

    Args:
        x (List[int]):
            Lengths of sentences.
        k (int):
            Number of clusters.
            This is an approximate value. The final number of clusters can be less or equal to k.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.

    Returns:
        centroids (List[float]):
            Average lengths in each cluster.
        clusters (List[List[int]]):
            List of clusters that hold indices of data points.
    """

    # the number of clusters must not be greater than the number of datapoints
    x, k = torch.tensor(x, dtype=torch.float), min(len(x), k)
    # collect unique datapoints
    d = x.unique()
    # initialize k centroids randomly
    c = d[torch.randperm(len(d))[:k]]
    # assign each datapoint to the cluster with the closest centroid
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)

    for _ in range(max_it):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster and move that the empty one
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        while len(none) > 0:
            for i in none:
                # the biggest cluster
                b = torch.where(mask[mask.sum(-1).argmax()])[0]
                # the datapoint farthest from the centroid of cluster b
                f = dists[b].argmax()
                # update the assigned cluster of f
                y[b[f]] = i
                # re-calculate the mask
                mask = torch.arange(k).unsqueeze(-1).eq(y)
            none = torch.where(~mask.any(-1))[0].tolist()
        # update the centroids
        c, old = (x * mask).sum(-1) / mask.sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)
        # stop iteration early if the centroids converge
        if c.equal(old):
            break
    # assign all datapoints to the new-generated clusters
    # the empty ones are discarded
    assigned = y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


def km_match(clusters):
    clusters = clusters.cpu()
    n_clusters = clusters.size(0)
    match = [None] * n_clusters
    label_pred = clusters.new_zeros(n_clusters)
    label_gold, _ = clusters.max(-2)

    def find_augmenting_path(v, S, T):
        S.add(v)
        for p in range(n_clusters):
            if label_gold[v] + label_pred[p] == clusters[p, v] and p not in T:
                T.add(p)
                if match[p] is None or find_augmenting_path(match[p], S, T):
                    match[p] = v
                    return True
        return False

    def update(S, T):
        a = min([
            label_gold[g] + label_pred[p] - clusters[p, g] for g in S
            for p in range(n_clusters) if p not in T
        ])
        for g in S:
            label_gold[g] -= a
        for p in T:
            label_pred[p] += a

    for g in range(n_clusters):
        while True:
            S, T = set(), set()
            has_find = find_augmenting_path(g, S, T)
            if not has_find:
                update(S, T)
            else:
                break
    correct = (label_gold + label_pred).sum()
    return correct, match


def tarjan(sequence):
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices that make up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)

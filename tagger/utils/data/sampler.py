import torch.distributed as dist
import torch.utils.data


class Sampler(torch.utils.data.Sampler):
    """
    Sampler supporting for bucketizing and token-level batchification.

    Args:
        buckets (Dict):
            The dict that maps each centroid to the indices of the clustering sentences.
            The centroid corresponds to the average length of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as batch_size.
        shuffle (bool, default: False):
            If True, the sampler will shuffle both buckets and samples in each bucket.
        distributed (bool, default: False):
            If True, the sampler will be used be used in conjunction with `torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
    """
    def __init__(self,
                 buckets,
                 batch_size,
                 seed=1,
                 shuffle=False,
                 distributed=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket)
                                         for size, bucket in buckets.items()])
        # number of chunks in each bucket, clipped by range [1, len(bucket)]
        self.chunks = [
            min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
            for size, bucket in zip(self.sizes, self.buckets)
        ]

        self.rank = dist.get_rank() if distributed else 0
        self.replicas = dist.get_world_size() if distributed else 1
        self.samples = sum(self.chunks) // self.replicas
        self.seed = seed if seed > 0 else (seed - 1)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(((self.epoch + 1) * self.seed) % 0x0fff_ffff_ffff_ffff)
        range_fn = torch.arange
        # if shuffle, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generte the same random sequence at each epoch
        if self.shuffle:
            range_fn = lambda x: torch.randperm(x, generator=g)

        total, count = 0, 0
        # TODO: more elegant way to deal with uneven data, which we directly discard right now
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                           for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                if count == self.samples:
                    break
                if total % self.replicas == self.rank:
                    count += 1
                    yield [self.buckets[i][j] for j in batch.tolist()]
                total += 1
        self.epoch += 1

    def __len__(self):
        return self.samples

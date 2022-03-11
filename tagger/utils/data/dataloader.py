import torch.utils.data


class DataLoader(torch.utils.data.DataLoader):
    """
    DataLoader, matching with the `Dataset` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield [f.compose(d) for f, d in batch.items()]

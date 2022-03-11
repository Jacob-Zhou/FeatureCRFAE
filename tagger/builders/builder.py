import os
import torch
import tagger
from tagger.utils.config import Config
from tagger.utils.data.dataset import Dataset
from tagger.utils.logging import init_logger, logger
from datetime import datetime, timedelta


class Builder(object):
    NAME = None
    FILE_NAME = "builder"

    def __init__(self, args, model, fields):
        self.args = args
        self.model = model
        self.fields = fields

    def evaluate(self, data, verbose=True, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.fields.train()
        logger.info("Loading the data")
        dataset = Dataset(self.fields, data)
        dataset.build(args.batch_size, args.n_buckets)
        logger.info(f"\n{dataset}")

        logger.info("Evaluating the dataset")
        start = datetime.now()
        loss, metric = self._evaluate(dataset.loader)
        elapsed = datetime.now() - start
        logger.info(f"Loss: {loss:>8.4f} - {metric}")
        logger.info(
            f"{elapsed}s elapsed, {len(dataset)/elapsed.total_seconds():.2f} Sents/s"
        )

        return loss, metric

    def predict(self, data, pred=None, verbose=True, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.fields.eval()

        logger.info("Loading the data")
        dataset = Dataset(self.fields, data)
        dataset.build(args.batch_size, args.n_buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        logger.info(f"Saving predicted results to {pred}")
        self.fields.save(pred, dataset.sentences)
        logger.info(
            f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s"
        )

        return dataset

    @classmethod
    def load(cls, path, **kwargs):
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a path to a directory containing a pre-trained builder, e.g., `./<path>/model`.
                - a filenane to a file containing a pre-trained builder, e.g., `./<path>/model/model.pth`.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations and initiate the model.

        Examples:
            >>> from tagger import Builder
            >>> builder = Builder.load('model')
            >>> builder = Builder.load('./model/model.pth')
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.isdir(path):
            if os.path.exists(path):
                file_name = os.path.join(path, cls.FILE_NAME + ".pth")
            else:
                raise RuntimeError("path don't exist")
        else:
            file_name = path
        if os.path.exists(file_name):
            state = torch.load(file_name, map_location=args.device)
        else:
            raise RuntimeError("path don't exist")

        cls = tagger.BUILDER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        fields = state['fields']
        return cls(args, model, fields)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        state = {
            'name': self.NAME,
            'args': args,
            'state_dict': state_dict,
            'fields': self.fields
        }
        torch.save(state, os.path.join(path, self.FILE_NAME + ".pth"))

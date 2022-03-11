import argparse

from tagger.builders.crf_ae_builder import CRFAEBuilder
from tagger.cmds.cmd import parse
from tagger.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Create first-order CRF Dependency Parser.')
    parser.set_defaults(Builder=CRFAEBuilder)

    parser.add_argument('--n-bottleneck',
                        default=5,
                        type=int,
                        help='max dim of bottle neck')
    parser.add_argument('--min-freq',
                        default=2,
                        type=int,
                        help='max num of buckets to use')
    parser.add_argument('--feat-min-freq',
                        default=50,
                        type=int,
                        help='max num of buckets to use')
    parser.add_argument('--n-layers',
                        default=None,
                        type=int,
                        help='layer of elmo to use in train CRF')
    parser.add_argument('--layer',
                        default=None,
                        type=int,
                        help='layer of elmo to use in train CRF')
    parser.add_argument('--encoder',
                        choices=['bert', 'elmo', 'lstm'],
                        help='choices of additional features')
    parser.add_argument('--plm', default=None, help='path to evaluate file')
    parser.add_argument('--ud-mode', action="store_true")
    parser.add_argument('--ud-feature', action="store_true")
    parser.add_argument('--replace-punct', action="store_true")
    parser.add_argument('--embed', default=None, help='path to evaluate file')
    parser.add_argument('--ignore-capitalized',
                        action="store_true",
                        help='random init CRF-AE model')
    parser.add_argument('--language-specific-strip',
                        action="store_true",
                        help='random init CRF-AE model')
    parser.add_argument('--language',
                        default="en",
                        help='path to evaluate file')
    parser.add_argument('--without-feature',
                        action="store_true",
                        help='do not use feature')
    parser.add_argument('--without-fd-repr',
                        action="store_true",
                        help='do not use finite difference repersent')

    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # arguments for train
    subparser = subparsers.add_parser('train', help='Train model.')
    subparser.add_argument('--train', help='path to train file')
    subparser.add_argument('--evaluate', help='path to evaluate file')
    subparser.add_argument('--test', default=None, help='path to test file')
    subparser.add_argument('--build',
                           '-b',
                           action="store_true",
                           help='whether to build the model first')
    subparser.add_argument('--rand-init',
                           action="store_true",
                           help='random init CRF-AE model')
    subparser.add_argument('--feature-hmm-path',
                           default=None,
                           help='path to evaluate file')
    subparser.add_argument('--hmm-epochs',
                           default=50,
                           type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--init-epochs',
                           default=5,
                           type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--epochs',
                           default=50,
                           type=int,
                           help='max num of buckets to use')

    # arguments for evaluate
    subparser = subparsers.add_parser(
        'evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--data', help='path to dataset')

    # arguments for predict
    subparser = subparsers.add_parser(
        'predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--data', help='path to dataset')
    subparser.add_argument('--pred', help='path to predicted result')
    parse(parser)


if __name__ == "__main__":
    main()

import os
from datetime import datetime

import numpy as np
import torch
from tagger.utils.config import Config
from tagger.utils.logging import init_logger, logger
from tagger.utils.parallel import init_device


def parse(parser):
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--device',
                        '-d',
                        default='-1',
                        help='ID of GPU to use')
    parser.add_argument('--seed',
                        '-s',
                        default=1,
                        type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--threads',
                        '-t',
                        default=16,
                        type=int,
                        help='max num of threads')
    parser.add_argument('--batch-size',
                        default=5000,
                        type=int,
                        help='batch size')
    parser.add_argument('--n-buckets',
                        default=32,
                        type=int,
                        help='max num of buckets to use')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help='node rank for distributed training')
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    Builder = args.pop('Builder')

    torch.set_num_threads(args.threads)
    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    init_device(args.device, args.local_rank)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().isoformat().split(".")[0]
    timestamp = str.replace(timestamp, ':', '-')
    args.timestamp = timestamp
    init_logger(logger, os.path.join(args.path,
                                     f"{args.mode}-{timestamp}.log"))
    logger.info('\n' + str(args))

    if args.mode == 'train':
        builder = Builder.build(**args)
        builder.train(**args)
    elif args.mode == 'evaluate':
        builder = Builder.load(args.path)
        builder.evaluate(**args)
    elif args.mode == 'predict':
        builder = Builder.load(args.path)
        builder.predict(**args)

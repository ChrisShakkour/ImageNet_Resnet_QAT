import logging
from .resnet import *

def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.dataloader.dataset == 'imagenet':
        if args.arch == 'resnet18':
            model = resnet18(pretrained=args.pre_trained)
        elif args.arch == 'resnet34':
            model = resnet34(pretrained=args.pre_trained)
        elif args.arch == 'resnet50':
            model = resnet50(pretrained=args.pre_trained)
        elif args.arch == 'resnet101':
            model = resnet101(pretrained=args.pre_trained)
        elif args.arch == 'resnet152':
            model = resnet152(pretrained=args.pre_trained)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    logger.info(msg)

    return model
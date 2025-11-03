from utils.args import *
from utils.logger import *
from utils.config_loader import *
from datasets.fake_dataset_loader import *
from models.models import *

def main():
    args = get_args()
    logger = init_logger(level=args.logging)
    cfg = load_yaml_config(args.config_file, logger)

    # Initialize data loader
    train_loader, val_loader = build_fake_loaders(cfg)
    logger.info('Dataset `%s` size:' % cfg.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader))
                )

        # Create the model
    model = create_model(cfg)

if __name__ == "__main__":
    main()
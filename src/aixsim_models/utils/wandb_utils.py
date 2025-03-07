import wandb
import logging

from torchtitan.metrics import WandBLogger


logger = logging.getLogger(__name__)


def wandb_update_config(metric_logger: WandBLogger, config_d: dict):
    if not isinstance(metric_logger, WandBLogger):
        logger.warning("Metric logger is not a WandBLogger, skipping config update")
        return
    assert isinstance(config_d, dict), "config must be a dictionary"

    metric_logger.wandb.config.update(config_d)
    logger.info("Updated WandB config")


def wandb_extract_and_update_tags(metric_logger: WandBLogger, args):
    from aixsim_models.llm.arg_manager import PreTrainArgs as PlainArgs
    from aixsim_models.bitflip.arg_manager import PreTrainArgs as BitflipArgs

    if not isinstance(metric_logger, WandBLogger):
        logger.warning("Metric logger is not a WandBLogger, skipping tag update")
        return

    if not isinstance(args, (PlainArgs, BitflipArgs)):
        logger.error(
            f"The PreTrainArgs object is not of the expected type. Consider updating function `extract_and_update_tags` in file {__file__}"
        )
        raise NotImplementedError(f"args is of unsupported type {type(args)}")

    tags = tuple()
    tags += (args.model.name, args.model.flavor)
    tags += (args.training.dataset,)

    if isinstance(args, PlainArgs):
        tags += ("transform:plain",)
    elif isinstance(args, BitflipArgs):
        tags += ("transform:bitflip",)
        tags += (f"transform-flavor:{args.transform.transform_flavor}",)
    else:
        raise NotImplementedError(f"args is of unsupported type {type(args)}")
    metric_logger.wandb.run.tags += tags
    logger.info(f"Updating WandB tags: {tags}")

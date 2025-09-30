import jsonargparse
from .train import MyTrain
from .test import MyTest
from .hpo import MyHpo
from .logger import get_logger
from .generator import MyGenerator
from .initializer import MyInitializer
from .optimizer import MyOptimizer
from .lr_scheduler import MyLrScheduler
from .early_stopping import MyEarlyStopping
from .dataset import MyDatasetAbstract
from .metric import MyMetricAbstract
from .model import MyModelAbstract


def get_train_parser():
    train_parser = jsonargparse.ArgumentParser(description="Train AI models.")
    train_parser.add_argument("--config", action="config")
    train_parser.add_class_arguments(theclass=MyTrain, nested_key="train")

    train_parser.add_function_arguments(
        function=get_logger,
        nested_key="logger",
    )
    train_parser.add_class_arguments(
        theclass=MyGenerator,
        nested_key="generator",
    )
    train_parser.add_class_arguments(
        theclass=MyInitializer,
        nested_key="initializer",
    )
    train_parser.add_class_arguments(
        theclass=MyOptimizer,
        nested_key="optimizer",
    )
    train_parser.add_class_arguments(
        theclass=MyLrScheduler,
        nested_key="lr_scheduler",
    )
    train_parser.add_class_arguments(
        theclass=MyEarlyStopping,
        nested_key="early_stopping",
    )
    train_parser.add_subclass_arguments(
        baseclass=MyDatasetAbstract,
        nested_key="dataset",
    )
    train_parser.add_argument(
        "--metric",
        nargs="+",
        type=MyMetricAbstract,
        required=True,
        enable_path=True,
    )
    train_parser.add_subclass_arguments(
        baseclass=MyModelAbstract,
        nested_key="model",
    )

    return train_parser


def get_test_parser():
    test_parser = jsonargparse.ArgumentParser(description="Test AI models.")
    test_parser.add_argument("--config", action="config")
    test_parser.add_class_arguments(theclass=MyTest, nested_key=None)

    return test_parser


def get_hpo_parser():
    hpo_parser = jsonargparse.ArgumentParser(description="Hpo AI models.")
    hpo_parser.add_argument("--config", action="config")
    hpo_parser.add_class_arguments(theclass=MyHpo, nested_key="hpo")
    hpo_parser.add_argument(
        "--train", action=jsonargparse.ActionParser(parser=get_train_parser())
    )

    return hpo_parser


def get_config() -> tuple[jsonargparse.ArgumentParser]:
    parser = jsonargparse.ArgumentParser(
        description="Arguments of AI models.",
    )
    subcommands = parser.add_subcommands(required=True, dest="subcommand")

    train_parser = get_train_parser()
    subcommands.add_subcommand(name="train", parser=train_parser)

    test_parser = get_test_parser()
    subcommands.add_subcommand(name="test", parser=test_parser)

    hpo_parser = get_hpo_parser()
    subcommands.add_subcommand(name="hpo", parser=hpo_parser)

    return parser, train_parser, test_parser, hpo_parser

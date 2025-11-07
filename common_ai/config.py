import jsonargparse
from .train import MyTrain
from .test import MyTest
from .hta import MyHta
from .hpo import MyHpo
from .logger import get_logger
from .generator import MyGenerator
from .initializer import MyInitializer
from .optimizer import MyOptimizer
from .lr_scheduler import MyLrScheduler
from .early_stopping import MyEarlyStopping
from .profiler import MyProfiler
from .dataset import MyDatasetAbstract
from .metric import MyMetricAbstract
from .model import MyModelAbstract
from .inference import MyInferenceAbstract
from .shap import MyShapAbstract


def get_train_parser() -> jsonargparse.ArgumentParser:
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
    train_parser.add_class_arguments(
        theclass=MyProfiler,
        nested_key="profiler",
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


def get_test_parser() -> jsonargparse.ArgumentParser:
    test_parser = jsonargparse.ArgumentParser(description="Test AI models.")
    test_parser.add_argument("--config", action="config")
    test_parser.add_class_arguments(theclass=MyTest, nested_key=None)

    return test_parser


def get_infer_parser() -> jsonargparse.ArgumentParser:
    infer_parser = jsonargparse.ArgumentParser(description="Infer AI models.")
    infer_parser.add_argument("--config", action="config")
    infer_parser.add_argument("--input", required=True, type=str, help="input file")
    infer_parser.add_argument("--output", required=True, type=str, help="output file")
    infer_parser.add_subclass_arguments(
        baseclass=MyInferenceAbstract, nested_key="inference"
    )
    infer_parser.add_argument(
        "--test", action=jsonargparse.ActionParser(parser=get_test_parser())
    )

    return infer_parser


def get_explain_parser() -> jsonargparse.ArgumentParser:
    explain_parser = jsonargparse.ArgumentParser(description="Explain AI models.")
    explain_parser.add_argument("--config", action="config")
    explain_parser.add_subclass_arguments(baseclass=MyShapAbstract, nested_key="shap")
    explain_parser.add_subclass_arguments(
        baseclass=MyInferenceAbstract, nested_key="inference"
    )
    explain_parser.add_argument(
        "--test", action=jsonargparse.ActionParser(parser=get_test_parser())
    )
    explain_parser.add_subclass_arguments(
        baseclass=MyDatasetAbstract, nested_key="dataset"
    )

    return explain_parser


def get_hta_parser() -> jsonargparse.ArgumentParser:
    hta_parser = jsonargparse.ArgumentParser(description="Hta AI models.")
    hta_parser.add_argument("--config", action="config")
    hta_parser.add_class_arguments(theclass=MyHta, nested_key=None)

    return hta_parser


def get_hpo_parser() -> jsonargparse.ArgumentParser:
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

    infer_parser = get_infer_parser()
    subcommands.add_subcommand(name="infer", parser=infer_parser)

    explain_parser = get_explain_parser()
    subcommands.add_subcommand(name="explain", parser=explain_parser)

    hta_parser = get_hta_parser()
    subcommands.add_subcommand(name="hta", parser=hta_parser)

    hpo_parser = get_hpo_parser()
    subcommands.add_subcommand(name="hpo", parser=hpo_parser)

    return (
        parser,
        train_parser,
        test_parser,
        infer_parser,
        explain_parser,
        hta_parser,
        hpo_parser,
    )

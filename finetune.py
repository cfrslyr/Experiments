import os
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision
from tqdm import tqdm
import yaml
import enum
from enum import Enum
import argparse
from tensorboardX import SummaryWriter
from collections import OrderedDict
from termcolor import colored
import tabulate

from models import Finetune
from utils.logging import get_num_parameter, human_format, DummySummaryWriter, sizeof_fmt
from utils.config import parse_cli_overides
from utils.learning_rate import warmup_cosine_learning_rate
from utils.accumulators import Mean, Max


# fmt: off
config = OrderedDict(
    dataset="Cifar10",
    model="pseudo_mhsa",
    load_checkpoint_file=None,
    no_cuda=False,

    # === OPTIMIZER ===
    optimizer_learning_rate=1e-3,
    optimizer_warmup_ratio=0.05,  # period of linear increase for lr scheduler
    optimizer_betas=(0.9, 0.95), 
    optimizer_weight_decay=1e-4,
    batch_size=100,
    num_epochs=800,
    seed=42,

    # === Encoder & Decoder ===
    encoder_model_dim = 256,
    encoder_layers = 6,

    decoder_model_dim = 192,
    decoder_layers = 6,
    droprate = 0.1,

    # === MHSA ===
    num_heads = 8,
    encoder_mlp_dim = 512,
    decoder_mlp_dim = 384,
    attn_droprate = 0,
    mlp_droprate = 0.1,
    
    # === Kark ===
    encoder_hidden_dim = 256,
    decoder_hidden_dim = 192,

    encoder_inner_ref_config = [None, "self"],
    decoder_inner_ref_config = [None, "self"],

    encoder_outer_ref_config = [None, "self"],
    decoder_outer_ref_config = [None, "self"],

    # === LOGGING ===
    only_list_parameters=False,
    num_keep_checkpoints=5,
    plot_attention_positions=False,
    output_dir="./output.tmp",
)
# fmt: on

output_dir = "./output.tmp"  # Can be overwritten by a script calling this


def main():
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """

    """
    Directory structure:

      output_dir
        |-- config.yaml
        |-- best.checkpoint
        |-- last.checkpoint
        |-- tensorboard logs...
    """

    global output_dir
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok = True)

    # save config in YAML file
    store_config()

    # create tensorboard writter
    writer = SummaryWriter(logdir=output_dir, max_queue=100, flush_secs=10)
    print(f"Tensorboard logs saved in '{output_dir}'")

    # Set the seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # We will run on CUDA if there is a GPU available
    device = torch.device("cuda" if not config["no_cuda"] and torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    train_loader, test_loader = get_dataset(test_batch_size=config["batch_size"])
    model = get_model(device)

    print_parameters(model)
    if config["only_list_parameters"]:
        print_flops(model)

    if config["load_checkpoint_file"] is not None:
        restore_checkpoint(config["load_checkpoint_file"], model, device)

    if config["only_list_parameters"]:
        exit()

    total_steps = config["num_epochs"] * len(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=config["optimizer_learning_rate"],
                                 weight_decay=config["optimizer_weight_decay"]
                                 )
    criterion = torch.nn.CrossEntropyLoss()

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = Max()
    checkpoint_every_n_epoch = None
    if config["num_keep_checkpoints"] > 0:
        checkpoint_every_n_epoch = max(1, config["num_epochs"] // config["num_keep_checkpoints"])
    else:
        checkpoint_every_n_epoch = 999999999999
    global_step = 0

    for epoch in range(config["num_epochs"]):
        print("Epoch {:03d}".format(epoch))
        model.train()

        # Keep track of statistics during training
        mean_train_accuracy = Mean()
        mean_train_loss = Mean()

        with tqdm(train_loader) as pbar:
            for i, (batch_x, batch_y) in enumerate(pbar):
                global_step += 1
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Warmup learning rate
                warmup_cosine_learning_rate(global_step, optimizer, config["optimizer_learning_rate"], config["optimizer_warmup_ratio"], total_steps)

                # Compute gradients for the batch
                optimizer.zero_grad()
                prediction = model(batch_x)
                classification_loss = criterion(prediction, batch_y)
                loss = classification_loss

                acc = accuracy(prediction, batch_y)

                loss.backward()

                # Do an optimizer step
                optimizer.step()

                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/classification-loss", classification_loss, global_step)
                writer.add_scalar("train/accuracy", acc, global_step)

                # Store the statistics
                mean_train_loss.add(loss.item(), weight=len(batch_x))
                mean_train_accuracy.add(acc.item(), weight=len(batch_x))

                pbar.set_postfix(loss=loss.item(), accuracy=acc.item(), lr=optimizer.param_groups[0]['lr'])


            # Log training stats
            log_metric(
                "accuracy", {"epoch": epoch, "value": mean_train_accuracy.value()}, {"split": "train"}
            )
            log_metric(
                "cross_entropy", {"epoch": epoch, "value": mean_train_loss.value()}, {"split": "train"}
            )
            log_metric("lr", {"epoch": epoch, "value": optimizer.param_groups[0]['lr']}, {})

        # Evaluation
        with torch.no_grad():
            model.eval()
            mean_test_accuracy = Mean()
            mean_test_loss = Mean()
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                acc = accuracy(prediction, batch_y)
                mean_test_loss.add(loss.item(), weight=len(batch_x))
                mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        log_metric(
            "accuracy", {"epoch": epoch, "value": mean_test_accuracy.value()}, {"split": "test"}
        )
        log_metric(
            "cross_entropy", {"epoch": epoch, "value": mean_test_loss.value()}, {"split": "test"}
        )
        writer.add_scalar("eval/classification_loss", mean_test_loss.value(), epoch)
        writer.add_scalar("eval/accuracy", mean_test_accuracy.value(), epoch)

        # Store checkpoints for the best model so far
        is_best_so_far = best_accuracy_so_far.add(mean_test_accuracy.value())
        if is_best_so_far:
            store_checkpoint("best.checkpoint", model, epoch, mean_test_accuracy.value())
        if epoch % checkpoint_every_n_epoch == 0:
            store_checkpoint("{:04d}.checkpoint".format(epoch), model, epoch, mean_test_accuracy.value())

    # Store a final checkpoint
    store_checkpoint(
        "final.checkpoint", model, config["num_epochs"] - 1, mean_test_accuracy.value()
    )
    writer.close()

    # Return the optimal accuracy, could be used for learning rate tuning
    return best_accuracy_so_far.value()

def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()

def log_metric(name, values, tags):
    """
    Log timeseries data with values formatted to 6 significant digits.
    Placeholder implementation.
    This function should be overwritten by any script that runs this as a module.
    """
    for key, value in values.items():
        formatted_values = {}
        if "epo" not in key:
            formatted_values[key] = f"{value:.6f}"
        else:
            formatted_values[key] = f"{value}"

    # Print with formatted values
    print("{name}: {values} ({tags})".format(name=name, values=formatted_values, tags=tags))

def get_dataset(test_batch_size=100, shuffle_train=True, data_root="./data"):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    if config["dataset"] == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif config["dataset"] == "Cifar100":
        dataset = torchvision.datasets.CIFAR100
    elif config["dataset"].startswith("/"):
        train_data = torch.load(config["dataset"] + ".train")
        test_data = torch.load(config["dataset"] + ".test")
        train_set = TensorDataset(train_data["data"], train_data["target"])
        test_set = TensorDataset(test_data["data"], test_data["target"])

        training_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=shuffle_train
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=test_batch_size, shuffle=False
        )

        return training_loader, test_loader
    else:
        raise ValueError("Unexpected value for config[dataset] {}".format(config["dataset"]))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

    train_set = dataset(root=data_root, train=True, download=True, transform=transform)
    test_set = dataset(root=data_root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader

def get_model(device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """

    model = {
        'vit': lambda: Finetune(config),
        'pseudo_mhsa': lambda: Finetune(config),
        'optimized_mhsa': lambda: Finetune(config),
        'gaussian': lambda: Finetune(config),
        'spline': lambda: Finetune(config),
        'general_vit': lambda: Finetune(config),
        'linear': lambda: Finetune(config)
    }[config["model"]]()

    # if device == torch.device("cuda"):
    #     print("Use DataParallel if multi-GPU")
    #     model = torch.nn.DataParallel(model)
    #     torch.backends.cudnn.benchmark = True

    return model.to(device)

def print_parameters(model):
    # compute number of parameters
    num_params, _ = get_num_parameter(model, trainable=False)
    num_bytes = num_params * 32 // 8  # assume float32 for all
    print(f"Number of parameters: {human_format(num_params)} ({sizeof_fmt(num_bytes)} for float32)")
    num_trainable_params, trainable_parameters = get_num_parameter(model, trainable=True)
    print("Number of trainable parameters:", human_format(num_trainable_params))

    if config["only_list_parameters"]:
        # Print detailed number of parameters
        print(tabulate.tabulate(trainable_parameters))

def print_flops(model):
    shape = None
    if config["dataset"] in ["Cifar10", "Cifar100"]:
        shape = (1, 3, 32, 32)
    else:
        print(f"Unknown dataset {config['dataset']} input size to compute # FLOPS")
        return

    try:
        from thop import profile # type: ignore
    except:
        print("Please `pip install thop` to compute # FLOPS")
        return

    model = model.train()
    input_data = torch.rand(*shape)
    num_flops, num_params = profile(model, inputs=(input_data, ))
    print("Number of FLOPS:", human_format(num_flops))

def store_config():
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(dict(config), f, sort_keys=False)

def store_checkpoint(filename, model, epoch, test_accuracy):
    """Store a checkpoint file to the output directory"""
    path = os.path.join(output_dir, filename)

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    # remove buffer from checkpoint
    # TODO should not hard code
    def keep_state_dict_keys(key):
        if "self.R" in key:
            return False
        return True

    time.sleep(
        1
    )  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {
            "epoch": epoch,
            "test_accuracy": test_accuracy,
            "model_state_dict": OrderedDict([
                (key, value) for key, value in model.state_dict().items() if keep_state_dict_keys(key)
            ]),
        },
        path,
    )

def restore_checkpoint(filename, model, device):
    """Load model from a checkpoint"""
    print("Loading model parameters from '{}'".format(filename))
    with open(filename, "rb") as f:
        checkpoint_data = torch.load(f, map_location=device)

    try:
        model_dict = checkpoint_data["model_state_dict"]
        state_dict = {k:v for k,v in model_dict.items() if ("encoder" in k or "embed" in k or "cls" in k)}
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(colored("Missing state_dict keys in checkpoint", "red"), e)
        print("Retry import with current model values for missing keys.")
        state = model.state_dict()
        state.update(state_dict)
        model.load_state_dict(state, strict=False)


if __name__ == "__main__":
    # if directly called from CLI (not as module)
    # we parse the parameters overides
    config = parse_cli_overides(config)
    main()
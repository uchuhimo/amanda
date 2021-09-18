import argparse
import os
import pickle
import sys

import amanda
import keras
from keras import optimizers
from keras.models import load_model

from examples.anomaly_detection.tensorflow.data import datasets
from examples.anomaly_detection.tensorflow.utils.utils import model_train
from examples.profile.tensorflow.profiler import Profiler


def get_dataset(dataset_name):
    data_name = dataset_name.split("_")[0]
    data = datasets[data_name.lower()]
    if data_name == "simplednn":
        choice = dataset_name.split("_")[-1]
        (x, y), (x_val, y_val) = data["load_data"](method=choice)
    else:
        (x, y), (x_val, y_val) = data["load_data"]()
    preprocess_func = data["preprocess"]
    data_set = {}
    data_set["x"] = preprocess_func(x)
    data_set["x_val"] = preprocess_func(x_val)
    if dataset_name == "cifar10" or dataset_name == "mnist":
        labels = 10
        data_set["y"] = keras.utils.to_categorical(y, labels)
        data_set["y_val"] = keras.utils.to_categorical(y_val, labels)
    elif data_name == "reuters":
        labels = 46
        data_set["y"] = keras.utils.to_categorical(y, labels)
        data_set["y_val"] = keras.utils.to_categorical(y_val, labels)
    elif data_name == "imdb":
        labels = 2
        data_set["y"] = keras.utils.to_categorical(y, labels)
        data_set["y_val"] = keras.utils.to_categorical(y_val, labels)
    else:
        data_set["y"] = y
        data_set["y_val"] = y_val
    return data_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OL Problem Demo Detection & Autorepair"
    )
    parser.add_argument(
        "--model_path",
        "-mp",
        default=sys.path[0] + "/demo_case/Gradient_Vanish_Case/model.h5",
        help="model path",
    )
    parser.add_argument(
        "--config_path",
        "-cp",
        default=sys.path[0] + "/demo_case/Gradient_Vanish_Case/config.pkl",
        help="training configurations path",
    )  #
    parser.add_argument("--check_interval", "-ci", default=3, help="detection interval")
    parser.add_argument(
        "--result_dir",
        "-rd",
        default=sys.path[0] + "/tmp/result_dir",
        help="The dir to store results",
    )
    parser.add_argument(
        "--log_dir",
        "-ld",
        default=sys.path[0] + "/tmp/log_dir",
        help="The dir to store logs",
    )
    parser.add_argument(
        "--new_issue_dir",
        "-nd",
        default=sys.path[0] + "/tmp/new_issue",
        help="The dir to store models with new problem in detection",
    )
    parser.add_argument(
        "--root_dir",
        "-rtd",
        default=sys.path[0] + "/tmp",
        help="The root dir for other records",
    )
    args = parser.parse_args()

    model = load_model(os.path.abspath(args.model_path))

    with open(os.path.abspath(args.config_path), "rb") as f:  # input,bug type,params
        training_config = pickle.load(f)
    opt_cls = getattr(optimizers, training_config["optimizer"])
    opt = opt_cls(**training_config["opt_kwargs"])
    batch_size = training_config["batchsize"]
    epoch = training_config["epoch"]
    loss = training_config["loss"]
    dataset = get_dataset(training_config["dataset"])
    if "callbacks" not in training_config.keys():
        callbacks = []
    else:
        callbacks = training_config["callbacks"]
    check_interval = "epoch_" + str(args.check_interval)

    save_dir = args.result_dir
    log_dir = args.log_dir
    new_issue_dir = args.new_issue_dir
    root_path = args.root_dir

    params = {
        "beta_1": 1e-3,
        "beta_2": 1e-4,
        "beta_3": 70,
        "gamma": 0.7,
        "zeta": 0.03,
        "eta": 0.2,
        "delta": 0.01,
        "alpha_1": 0,
        "alpha_2": 0,
        "alpha_3": 0,
        "Theta": 0.6,
    }

    profiler = Profiler()
    with amanda.tool.apply(profiler):
        train_result, _, _ = model_train(
            model=model,
            train_config_set=training_config,
            optimizer=opt,
            loss=loss,
            dataset=dataset,
            iters=epoch,
            batch_size=batch_size,
            callbacks=callbacks,
            verb=1,
            checktype=check_interval,
            autorepair=True,
            save_dir=save_dir,
            determine_threshold=1,
            params=params,
            log_dir=log_dir,
            new_issue_dir=new_issue_dir,
            root_path=root_path,
        )

    profiler.export_chrome_trace("tmp/profile/autotrainer.json")

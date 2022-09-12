import argparse
import csv
import datetime
from gc import callbacks
import json
import os
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf
from preprocess import load_dataset
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from transformers import AutoTokenizer, TFAutoModel, TFPreTrainedModel

def create_model(config) -> tf.keras.Model:
    bert = TFAutoModel.from_pretrained(config["model"], from_pt=False)
    token_ids = tf.keras.layers.Input((config["max_len"],), dtype=np.int32)
    attention_mask = tf.keras.layers.Input((config["max_len"],), dtype=np.int32)
    inputs = [token_ids, attention_mask]
    if config["use_features"]:
        features = tf.keras.layers.Input((config["num_features"],), dtype=np.float32)
        inputs.append(features)
    emb = bert(input_ids=token_ids, attention_mask=attention_mask)
    if config["pooler"]:
        emb = emb.pooler_output
    else:
        emb = tf.keras.layers.GlobalAveragePooling1D()(emb.last_hidden_state, mask=attention_mask)
    if config["reduce"] > 0:
        emb = tf.keras.layers.Dense(config["reduce"], name="reduce", activation="tanh")(emb)
    if config["use_features"]:
        merged = tf.keras.layers.concatenate([emb, features], axis=-1)
    else:
        merged = emb
    hidden = tf.keras.layers.Dense(config["hidden"], activation="tanh")(merged)
    output = tf.keras.layers.Dense(1, activation=None)(hidden)
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=[output])
    return model

def train(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    train_texts, train_features, train_values = load_dataset(os.path.join(config["root"], "train.csv"))
    train_values = np.array(train_values)
    val_texts, val_features, val_values = load_dataset(os.path.join(config["root"], "val.csv"))
    val_values = np.array(val_values)
    processed = tokenizer(train_texts, padding="max_length", truncation=True, max_length=config["max_len"], return_tensors="np")
    train_inputs = [
        processed["input_ids"],
        processed["attention_mask"],
    ]
    processed = tokenizer(val_texts, padding="max_length", truncation=True, max_length=config["max_len"], return_tensors="np")
    val_inputs = [
        processed["input_ids"],
        processed["attention_mask"],
    ]
    if config["use_features"]:
        train_inputs.append(np.array(train_features))
        val_inputs.append(np.array(val_features))
    config["num_features"] = len(train_features[0])
    model = create_model(config)
    for layer in model.layers:
        if isinstance(layer, TFPreTrainedModel):
            layer.trainable = False
        if layer.name == "reduce" and not config["trainable_reduce"]:
            layer.trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="mse")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=3, restore_best_weights=True)
    history = model.fit(train_inputs, train_values, batch_size=16, callbacks=[callback],
        epochs=20, validation_data=(val_inputs, val_values), verbose=0)
    initial_epoch = int(np.argmin(history.history["val_loss"]))
    if config["finetune_bert"]:
        for layer in model.layers:
            if isinstance(layer, TFPreTrainedModel):
                layer.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        model.compile(optimizer=optimizer, loss="mse")
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=3, restore_best_weights=True)
        history = model.fit(train_inputs, train_values, batch_size=12, callbacks=[callback],
            epochs=20, validation_data=(val_inputs, val_values), verbose=0)
        finetune_epoch = int(np.argmin(history.history["val_loss"]))
        return {
            "loss": history.history["val_loss"][finetune_epoch],
            "initial_epochs": initial_epoch + 1,
            "finetune_epochs": finetune_epoch + 1,
        }
    else:
        return {
            "loss": history.history["val_loss"][initial_epoch],
            "initial_epochs": initial_epoch + 1,
            "finetune_epochs": 0,
        }

def train_and_save(config) -> float:
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    train_texts, train_features, train_values = load_dataset(os.path.join(config["root"], "train.csv"))
    val_texts, val_features, val_values = load_dataset(os.path.join(config["root"], "val.csv"))
    train_texts += val_texts
    train_features += val_features
    train_values += val_values
    train_values = np.array(train_values)
    test_texts, test_features, test_values = load_dataset(os.path.join(config["root"], "test.csv"))
    test_values = np.array(test_values)
    processed = tokenizer(train_texts, padding="max_length", truncation=True, max_length=config["max_len"], return_tensors="np")
    train_inputs = [
        processed["input_ids"],
        processed["attention_mask"],
    ]
    processed = tokenizer(test_texts, padding="max_length", truncation=True, max_length=config["max_len"], return_tensors="np")
    test_inputs = [
        processed["input_ids"],
        processed["attention_mask"],
    ]
    if config["use_features"]:
        train_inputs.append(np.array(train_features))
        test_inputs.append(np.array(test_features))
    config["num_features"] = len(train_features[0])
    model = create_model(config)
    for layer in model.layers:
        if isinstance(layer, TFPreTrainedModel):
            layer.trainable = False
        if layer.name == "reduce" and not config["trainable_reduce"]:
            layer.trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="mse")
    model.fit(train_inputs, train_values, batch_size=16,
        epochs=config["initial_epochs"], verbose=0)
    if config["finetune_bert"]:
        for layer in model.layers:
            if isinstance(layer, TFPreTrainedModel):
                layer.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        model.compile(optimizer=optimizer, loss="mse")
        model.fit(train_inputs, train_values, batch_size=12,
            epochs=config["finetune_epochs"], verbose=0)
    mse = model.evaluate(test_inputs, test_values, verbose=0)
    model.save(os.path.join(config["root"], "model"))
    config["mse"] = mse
    with open(os.path.join(config["root"], "config.json"), "wt") as f:
        json.dump(config, f)
    return mse

def hyperparameter_tuning(folder: str, gpu: bool, minutes: int) -> float:
    config = {
        "root": folder,
        "model": tune.choice(["readerbench/RoBERT-base","readerbench/RoBERT-small", "readerbench/RoBERT-large"]),
        "max_len": 256,
        "use_features": tune.choice([True, False]),
        "pooler": tune.choice([True, False]),
        "reduce": tune.choice([0, 32, 64, 128, 256]),
        "hidden": tune.choice([32, 64, 128, 256]),
        "finetune_bert": tune.choice([True, False]),
        "trainable_reduce": tune.choice([True, False]),
        "lr": 2e-5,
    }
    reporter = tune.CLIReporter(max_progress_rows=10, max_report_frequency=60)
    optuna_search = OptunaSearch(
        metric="loss",
        mode="min")
    analysis = tune.run(
        train, 
        metric="loss",
        mode="min",
        time_budget_s=datetime.timedelta(minutes=minutes),
        config=config, 
        num_samples=-1, 
        resources_per_trial={"cpu": 2, "gpu": int(gpu)},
        reuse_actors=True,
        progress_reporter=reporter,
        search_alg=optuna_search,
    )
    best_config = analysis.best_config
    best_config["initial_epochs"] = analysis.best_result["initial_epochs"]
    best_config["finetune_epochs"] = analysis.best_result["finetune_epochs"]
    return train_and_save(best_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument("folder", type=str, help="Dataset folder")
    parser.add_argument("--gpu", action='store_true', help="GPU is available")
    parser.add_argument("--minutes", default=120, type=int, help="Total timelimit (in minutes) for the experiment")
    
    args = parser.parse_args()
    
    hyperparameter_tuning(os.path.abspath(args.folder), args.gpu, args.minutes)
    
    
    
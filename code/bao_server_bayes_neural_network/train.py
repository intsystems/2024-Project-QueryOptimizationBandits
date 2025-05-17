from time import time
import storage
import model
import os
import shutil
import reg_blocker
from itertools import product
import torch
import numpy as np

class BaoTrainingException(Exception):
    pass

def train_and_swap(fn, old, tmp, verbose=False):
    if os.path.exists(fn):
        old_model = model.BaoRegression(have_cache_data=True)
        old_model.load(fn)
    else:
        old_model = None

    new_model = train_and_save_model(tmp, verbose=verbose)
    max_retries = 5
    current_retry = 1
    while not reg_blocker.should_replace_model(old_model, new_model):
        if current_retry >= max_retries == 0:
            print("Could not train model with better regression profile.")
            return
        
        print("New model rejected when compared with old model. "
              + "Trying to retrain with emphasis on regressions.")
        print("Retry #", current_retry)
        new_model = train_and_save_model(tmp, verbose=verbose,
                                         emphasize_experiments=current_retry)
        current_retry += 1

    if os.path.exists(fn):
        shutil.rmtree(old, ignore_errors=True)
        os.rename(fn, old)
    os.rename(tmp, fn)

def train_and_save_model(fn, verbose=True, emphasize_experiments=0):
    all_experience = storage.experience()

    for _ in range(emphasize_experiments):
        all_experience.extend(storage.experiment_experience())
    
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]        
    
    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    reg = model.BaoRegression(have_cache_data=True, verbose=verbose)
    reg.fit(x, y)
    reg.save(fn)
    return reg

def train_and_save_models(fn, verbose=True, emphasize_experiments=0):
    if verbose:
        print("Start training Bao")
    all_experience = storage.experience()

    for _ in range(emphasize_experiments):
        all_experience.extend(storage.experiment_experience())
    
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]        
    
    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    lrs = [0.001, 0.0005, 0.0001]
    betas = [0.01, 0.005, 0.001]

    val_size = int(0.2 * len(x))

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    val_indices = indices[:val_size]
    
    x_val = np.array(x)[val_indices]
    y_val = np.array(y)[val_indices]

    best_score = float("inf")

    best_model = None
    start = time()

    best_parameters = None

    for lr, beta in product(lrs, betas):
        if verbose:
            print(f"Learning rate: {lr}, beta: {beta}")
        reg = model.BaoRegression(have_cache_data=True, verbose=verbose)
        reg.fit(x, y, lr=lr, beta=beta)
        y_pred = np.reshape(reg.predict(x_val), len(x_val))
        current_score = ((y_val - y_pred)**2).mean()
        if verbose:
            print(f"Validation score: {(current_score*(1e-6)):.3f}")
        if current_score < best_score:
            reg.save(fn)
            best_model = reg
            best_score = current_score
            best_parameters = (lr, beta)

    stop = time()
    if verbose:
        print(f"Training took {int((stop - start) * 1000)} ms")
        print(f"Best parameters: lr / beta {best_parameters[0]} / {best_parameters[1]}")
    return best_model


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: train.py MODEL_FILE")
        exit(-1)
    train_and_save_model(sys.argv[1])

    print("Model saved, attempting load...")
    reg = model.BaoRegression(have_cache_data=True)
    reg.load(sys.argv[1])


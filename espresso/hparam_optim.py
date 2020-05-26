#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser(
    description='Tunes parameters of a script.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--space',
    required=True,
    type=str,
    help='Path to a .py space file.',
)
parser.add_argument(
    '--train-command',
    required=True,
    type=str,
    help="Command that will train the model."
)
parser.add_argument(
    '--train-arguments',
    type=str,
    help="Train command arguments",
    default=''
)
parser.add_argument(
    '--eval-command',
    required=True,
    type=str,
    help="Command (and additional args) that will evaluate the model. Should return only a value."
)
parser.add_argument(
    '--maximize',
    action='store_true',
    help="If set, maximizes the evaluation metric instead of minimizing."
)

args = parser.parse_args()
space_file = os.path.splitext(os.path.basename(args.space))[0]

##

import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt.mongoexp import MongoTrials
import subprocess


import importlib.machinery
loader = importlib.machinery.SourceFileLoader(space_file, args.space)
mod = loader.load_module()

def exec(command):
    return subprocess.check_output(command.split(' '))

def parse_params(params_dict):
    parsed = ""
    for arg in params_dict:
        if arg == 'run_id':
            continue
        parsed += " --{} {}".format(arg, params_dict[arg])
    return parsed

def parse_runid(text, run_id):
    return text.replace('{{RUN_ID}}', run_id)

def objective_func(hparams):
    if 'run_id' in hparams:
        run_id = str(hparams['run_id']).replace('.', '')

        args.train_arguments = parse_runid(args.train_arguments, run_id)
        args.train_command = parse_runid(args.train_command, run_id)
        args.eval_command = parse_runid(args.eval_command, run_id)

    parsed_params = parse_params(hparams)

    # train
    exec(args.train_command + ' ' + args.train_arguments + parsed_params)

    # eval
    result = exec(args.eval_command)
    result = float(result)

    if args.maximize:
        result = -result

    return result
##

space = mod.SpaceParams.get_space()
trials = MongoTrials(
    'mongo://skinner:1234/foo_db/jobs',
    exp_key='exp11',
    workdir=os.getcwd()
)
best = fmin(
    objective_func,
    space,
    algo=tpe.suggest,
    catch_eval_exceptions=True,
    verbose=True,
    max_queue_len=20,
    max_evals=40,
    trials=trials
)

print("\n\nOn a fini")

print("a")
print(best)
print("c")

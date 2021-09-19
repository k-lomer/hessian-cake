import tensorflow as tf
import os
from datetime import datetime

class Logger:
    def __init__(self, root_logdir=None, profile=False, metrics=[]):
        self.profile = profile
        self.metrics = list(metrics) + ["test_loss", "sweeps", "time"]
        self.values = {m: None for m in self.metrics}
        self.values["time"] = 0
        self.values["sweeps"] = 0
        self.clock_time = 0

        if root_logdir is None:
            root_logdir = "logs/active/"
        self.logdir = root_logdir + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        os.mkdir(self.logdir)

        with open(self.logdir + "batch.out", "a") as f:
            f.write(",".join(self.metrics) + "\n")

        with open(self.logdir + "epoch.out", "a") as f:
            f.write(",".join(self.metrics) + "\n")


    def write_metrics(self, f):
        f.write(",".join(str(self.values[m]) for m in self.metrics))
        f.write("\n")

    def log_value(self, metric, value, accum=False):
        if metric in self.metrics:
            if accum:
                self.values[metric] += value
            else:
                self.values[metric] = value

    def write_log(self, batch=False, epoch=False):
        if batch:
            with open(self.logdir + "batch.out", "a") as f:
                self.write_metrics(f)
        if epoch:
            with open(self.logdir + "epoch.out", "a") as f:
                self.write_metrics(f)


    def start_clock(self):
        self.clock_time = datetime.now()

    def stop_clock(self):
        self.values["time"] += (datetime.now() - self.clock_time).total_seconds()

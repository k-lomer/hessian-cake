import matplotlib
import platform
import matplotlib.pyplot as plt
import json
import os.path

platform = platform.system()
if platform == "Linux":
    matplotlib.use('Agg')

def plot_metric(run_file, metrics):
    """
    plot the given metrics per iteration and
    :param run_file: string: path of JSON file containing run values
    :param metrics: list(string): the metrics to plot
    """
    file_name = os.path.splitext(run_file)[0]
    with open(run_file, 'r') as f:
        run_data = json.load(f)

    for metric in metrics:
        for run in run_data["runs"]:
            plt.plot(run[metric], label=run["name"])

        #plt.legend()
        plt.yscale("log")
        plt.ylabel(metric)
        plt.xlabel("iterations")
        plt.title(run_data["name"])
        plt.savefig(file_name + "_" + metric + ".png")
        if platform != "Linux":
            plt.show()
        plt.clf()




def plot_comparison(run_file, average=False, min_limits=False):
    """
    plot the different runs with respect to iterations, sweeps and time
    write to jpg file
    :param run_file: string: path of JSON file containing run values
    :param average: bool: whether this is from a single test or an average across many
    """
    file_name = os.path.splitext(run_file)[0]
    with open(run_file, 'r') as f:
        run_data = json.load(f)

    for measure in ["iter", "sweeps", "time"]:
        for run in run_data["runs"]:
            if average:
                plt.plot(run[measure], label=run["name"])
            elif measure == "iter":
                plt.plot(run["loss"], label=run["name"])
            else:
                plt.plot(run[measure], run["loss"], label=run["name"])

        if min_limits and average:
            x_limit = min(len(run[measure]) for run in run_data["runs"])
            plt.xlim([0,x_limit])

        plt.legend()
        plt.yscale("log")
        plt.ylabel("loss")
        plt.xlabel(measure)
        #plt.title(run_data["name"])
        plt.savefig(file_name + "_" + measure + ".png")
        if platform != "Linux":
            plt.show()
        plt.clf()

if __name__ == "__main__":
    plot_comparison('./../../runs/parameter_comparison/ew/EW_final_comparison.json', average=True, min_limits=True)


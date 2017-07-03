import os
import argparse
import datetime
from itertools import groupby
import matplotlib.pyplot as plt


def show_results(log_file):
    with open(log_file, 'r') as f:
        contents = f.read().splitlines()
    contents = [list(group) for k, group in groupby(contents, lambda x: x == "-") if not k]

    batch_data = []
    for batch in contents:
        batch_data_norm = dict()
        for entry in batch:
            key, value = entry.split(" ")
            if key not in batch_data_norm:
                batch_data_norm[key] = []
            batch_data_norm[key].append(float(value))
        batch_data.append(batch_data_norm)

    fig, ax1 = plt.subplots()
    fig.suptitle('Training/Validaton Losses')
    ax1.plot( [y for x in batch_data for y in x['training_loss']],'-r', label='Training loss')
    ax1.plot( [y for x in batch_data for y in x['validation_loss']],'-g', label='Validation loss')
    ax1.legend(loc='upper left')

    fig, ax2 = plt.subplots()
    fig.suptitle('Training Losses per batch')
    ax2.plot( [y for x in batch_data for y in x['training_batch_loss']], label='Training loss (per batch)')
    ax2.legend(loc='upper left')

    plt.legend(loc='upper left')
    plt.show()

def get_most_recent_log():
    onlyfiles = [f for f in os.listdir("logs") if os.path.isfile(os.path.join("logs", f)) and os.path.splitext(f)[1] ==".txt"]
    file_dates = [ datetime.datetime.strptime(os.path.splitext(f)[0], '%Y-%m-%d-%H-%M-%S') for f in onlyfiles ]

    return os.path.join("logs", onlyfiles[file_dates.index(max(file_dates))] )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", action="store",
                        help="Log file name (if not specified will use most recent")

    args = parser.parse_args()
    if args.file_name is not None:
        log_file = os.path.join("logs", args.file_name)
        print("Using specified log file in {}".format(log_file))
    else:
        log_file = get_most_recent_log()
        print("Using most recent log file in {}".format(log_file))

    show_results(log_file)



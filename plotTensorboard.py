from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import glob
import numpy as np


def moving_average(a, n=20):
    """Compute moving average of array `a` with window size `n`."""
    if len(a) < n:
        return a  # Return as is if not enough data points
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

for exp in glob.glob('summaries/model_v3'):
    print(f"Processing experiment: {exp}")

    event_files = glob.glob(exp + '*/event*')
    if not event_files:
        print(f"No event files found in {exp}")
        continue
    for file in event_files:
    # file = event_files[0]
        print(f"Using event file: {file}")
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        event_acc = EventAccumulator(file)
        event_acc.Reload()

        # Check available tags
        available_tags = event_acc.Tags()['scalars']
        print(f"Available scalar tags: {available_tags}")
        print(len(event_acc.Scalars('Accuracy/test_accuracy')))
        # Process 'Accuracy/test_accuracy'
        if 'Accuracy/test_accuracy' in available_tags:
            accuracy_scalars = event_acc.Scalars('Accuracy/test_accuracy')
            if accuracy_scalars:
                steps, wall_times, values = zip(*[(s.step, s.wall_time, s.value) for s in accuracy_scalars])
                # axs[0].plot(moving_average(np.array(values)), label=exp.split('/')[-1])
                axs[0].plot(values, label=exp.split('/')[-1])
                axs[0].legend()
                axs[0].set_title("Accuracy/test_accuracy")
            else:
                print("No scalar data found for Accuracy/test_accuracy")
        else:
            print("Tag 'Accuracy/test_accuracy' not found in event file.")

        # Process 'Loss/test_loss'
        if 'Loss/test_loss' in available_tags:
            loss_scalars = event_acc.Scalars('Loss/test_loss')
            if loss_scalars:
                steps, wall_times, values = zip(*[(s.step, s.wall_time, s.value) for s in loss_scalars])
                # axs[1].plot(moving_average(np.array(values)), label=exp.split('/')[-1])
                axs[1].plot(values, label=exp.split('/')[-1])
                axs[1].legend()
                axs[1].set_title("Loss/test_loss")
            else:
                print("No scalar data found for Loss/test_loss")
        else:
            print("Tag 'Loss/test_loss' not found in event file.")

        plt.show()
plt.close()

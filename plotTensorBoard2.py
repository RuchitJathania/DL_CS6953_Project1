import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plotAndSave(save_dir, summary_dir):
    # Directory to save plots
    # save_dir = "checkpoints/resnet4_v2"
    os.makedirs(save_dir, exist_ok=True)

    plot_index = 1  # Initialize plot index

    for exp in glob.glob(summary_dir):
        print(f"Processing experiment: {exp}")

        event_files = glob.glob(exp + '*/event*')
        if not event_files:
            print(f"No event files found in {exp}")
            continue

        for file in event_files:

            event_acc = EventAccumulator(file)
            event_acc.Reload()

            # Check available tags
            available_tags = event_acc.Tags()['scalars']

            if 'Accuracy/test_accuracy' not in available_tags:
                continue

            test_accuracy_scalars = event_acc.Scalars('Accuracy/test_accuracy')

            if len(test_accuracy_scalars) <= 1:
                print("Skipping saving plot as Accuracy/test_accuracy has 5 or fewer points.")
                continue
            print(f"Using event file: {file}")
            print(f"Available scalar tags: {available_tags}")
            print(len(test_accuracy_scalars))
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

            # Process 'Accuracy/test_accuracy'
            if test_accuracy_scalars:
                steps, wall_times, values = zip(*[(s.step, s.wall_time, s.value) for s in test_accuracy_scalars])
                axs[0].plot(values, label=exp.split('/')[-1])
                axs[0].legend()
                axs[0].set_title("Accuracy/test_accuracy")
            else:
                print("No scalar data found for Accuracy/test_accuracy")

            # Process 'Loss/test_loss'
            if 'Loss/test_loss' in available_tags:
                loss_scalars = event_acc.Scalars('Loss/test_loss')
                if loss_scalars:
                    steps, wall_times, values = zip(*[(s.step, s.wall_time, s.value) for s in loss_scalars])
                    axs[1].plot(values, label=exp.split('/')[-1])
                    axs[1].legend()
                    axs[1].set_title("Loss/test_loss")
                else:
                    print("No scalar data found for Loss/test_loss")
            else:
                print("Tag 'Loss/test_loss' not found in event file.")

            # Save plot
            plot_path = os.path.join(save_dir, f"plot_{plot_index}.png")
            plt.savefig(plot_path)
            print(f"Saved plot as {plot_path}")
            plt.show()
            plot_index += 1  # Increment plot index
            plt.close(fig)  # Close figure to free memory

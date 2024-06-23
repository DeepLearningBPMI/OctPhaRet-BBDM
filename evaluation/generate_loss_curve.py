from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    log_file_paths = {'s=1' : '/scistor/guest/mhn744/BScProject/BBDM/results/dataset_cropped_clipped/LBBDM-f8/log/events.out.tfevents.1717179065.node013.1145177.0',
                    's=0.5': '/scistor/guest/mhn744/BScProject/BBDM/results/dataset_cropped_clipped_max_var_05/LBBDM-f8/log/events.out.tfevents.1717179222.node015.117045.0',
                    's=0.2': '/scistor/guest/mhn744/BScProject/BBDM/results/dataset_cropped_clipped_max_var_02/LBBDM-f8/log/events.out.tfevents.1717179307.node010.1893620.0'}

    plt.figure()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss progression during training')
    for s, log_file_path in log_file_paths.items():
        ea = event_accumulator.EventAccumulator(log_file_path)
        ea.Reload()  # Load events from the file

        # Extract values for training loss
        train_loss_events = ea.Scalars('loss/train')

        # Extract steps and values for training loss
        train_steps = [event.step for event in train_loss_events]
        train_values = [event.value for event in train_loss_events]

        # Plot training loss
        train_epochs = np.arange(100)
        train_values_averaged = np.empty(100)
        for i in range(100):
            train_values_averaged[i] = np.mean(train_values[i*len(train_values)//100:(i+1)*len(train_values)//100])
        plt.plot(train_epochs, train_values_averaged, label=f'Training loss ({s})')

    # save figure
    plt.legend()
    plt.savefig('/scistor/guest/mhn744/BScProject/BBDM/evaluation_results/loss_curves.png')

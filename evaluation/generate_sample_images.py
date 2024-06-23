import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


if __name__ == '__main__':
    sampling_steps = [20, 50, 100, 200]
    max_vars = {0.2: 'dataset_cropped_clipped_max_var_02',
                0.5: 'dataset_cropped_clipped_max_var_05',
                1: 'dataset_cropped_clipped'}

    # For each max. variance, create a 4 x 3 grid of images showing the condition, ground truth, and 1 prediction for five random images
    for max_var, max_var_path in max_vars.items():
        fig, axs = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle(f'Sample condition, ground truth and prediction images. Scaling parameter ($s$): {max_var}.\n\n', fontsize=20)

        # fig.tight_layout(rect=[0, 0, 1, 0.95])
        axs[0, 0].set_title('Condition', fontsize=20)
        axs[0, 1].set_title('Ground Truth', fontsize=20)
        axs[0, 2].set_title('Prediction', fontsize=20)
        axs[0, 0].set_ylabel('20 Sampling Steps', fontsize=20)
        axs[1, 0].set_ylabel('50 Sampling Steps', fontsize=20)
        axs[2, 0].set_ylabel('100 Sampling Steps', fontsize=20)
        axs[3, 0].set_ylabel('200 Sampling Steps', fontsize=20)

        plt.subplots_adjust(hspace=0.2)
        plt.subplots_adjust(wspace=0.2)



        # Make sure there is not too much space at the top and bottom of the plot, as well as at the left and right

        for i, sampling_step in enumerate(sampling_steps):
            pred_path = f'/scistor/guest/mhn744/BScProject/BBDM/results/{max_var_path}/LBBDM-f8/sample_to_eval/{sampling_step}'
            condition_path = f'/scistor/guest/mhn744/BScProject/BBDM/results/{max_var_path}/LBBDM-f8/sample_to_eval/condition'
            ground_truth_path = f'/scistor/guest/mhn744/BScProject/BBDM/results/{max_var_path}/LBBDM-f8/sample_to_eval/ground_truth'

            img_path = np.random.choice(os.listdir(pred_path))

            # Read images
            pred_imgs = [Image.open(os.path.join(pred_path, img_path, f'output_{i}.png')).convert('L') for i in range(5)]
            condition_img = Image.open(os.path.join(condition_path, img_path) + '.png').convert('L')
            ground_truth_img = Image.open(os.path.join(ground_truth_path, img_path) + '.png').convert('L')
            
            pred_imgs = [np.array(img) for img in pred_imgs]
            condition_img = np.array(condition_img)
            ground_truth_img = np.array(ground_truth_img)

            # Display images
            axs[i, 0].imshow(condition_img, cmap='gray')
            axs[i, 1].imshow(ground_truth_img, cmap='gray')
            axs[i, 2].imshow(pred_imgs[0], cmap='gray')
            
            # Remove axis tickers, but not labels
            axs[i, 0].set_xticks([])
            axs[i, 0].set_yticks([])
            axs[i, 1].set_xticks([])
            axs[i, 1].set_yticks([])
            axs[i, 2].set_xticks([])
            axs[i, 2].set_yticks([])

        fig.tight_layout()
        plt.savefig(f'/scistor/guest/mhn744/BScProject/BBDM/evaluation_results/sample_images_{max_var}.png')
        plt.close()

            

import numpy as np
from tqdm import tqdm
from PIL import Image
import os

def correlation_and_mse(pred_img, ground_truth_img, mask):
    """Calculate the correlation and mean squared error between the predicted and ground truth images in a zone of interest specified by the mask."""
    pred_img = pred_img.flatten()
    ground_truth_img = ground_truth_img.flatten()
    mask = mask.flatten()

    # Make new arrays with only pixels in the mask
    pred_img = pred_img[mask != 0]
    ground_truth_img = ground_truth_img[mask != 0]

    # # Fit linear regression line
    # m, b = np.polyfit(pred_img, ground_truth_img, 1)

    # # Make scatter plot with small dots and linear regression line fitted
    # plt.figure()
    # plt.scatter(pred_img, ground_truth_img, s=1)
    # plt.plot(pred_img, m*pred_img + b, color='red')
    # plt.xlabel('Predicted')
    # plt.ylabel('Ground truth')
    # plt.title('Predicted vs ground truth')
    # plt.savefig('scatter_plot.png')

    # Calculate pearson correlation
    corr = np.corrcoef(pred_img, ground_truth_img)[0, 1]
    mse = np.mean((pred_img - ground_truth_img)**2)
    return corr, mse


if __name__ == '__main__':
    sampling_steps = {20, 50, 100, 200}
    max_vars = {0.2: 'dataset_cropped_clipped_max_var_02',
                0.5: 'dataset_cropped_clipped_max_var_05',
                1: 'dataset_cropped_clipped'}

    output_path = '/scistor/guest/mhn744/BScProject/BBDM/evaluation_results'

    for sampling_step in sampling_steps:
        for max_var, max_var_path in max_vars.items():
            pred_path = f'/scistor/guest/mhn744/BScProject/BBDM/results/{max_var_path}/LBBDM-f8/sample_to_eval/{sampling_step}'
            condition_path = f'/scistor/guest/mhn744/BScProject/BBDM/results/{max_var_path}/LBBDM-f8/sample_to_eval/condition'
            ground_truth_path = f'/scistor/guest/mhn744/BScProject/BBDM/results/{max_var_path}/LBBDM-f8/sample_to_eval/ground_truth'

            correlation_values = np.empty((len(os.listdir(pred_path)), 5))
            mse_values = np.empty((len(os.listdir(pred_path)), 5))

            for i, img_path in tqdm(enumerate(os.listdir(pred_path)), desc='Processing images'):

                # Read images
                pred_imgs = [Image.open(os.path.join(pred_path, img_path, f'output_{i}.png')).convert('L') for i in range(5)]
                condition_img = Image.open(os.path.join(condition_path, img_path) + '.png').convert('L')
                ground_truth_img = Image.open(os.path.join(ground_truth_path, img_path) + '.png').convert('L')
                
                pred_imgs = [np.array(img) for img in pred_imgs]
                condition_img = np.array(condition_img)
                ground_truth_img = np.array(ground_truth_img)

                # Create mask
                mask = np.load('/scistor/guest/mhn744/BScProject/BBDM/evaluation_results/mask_ring_of_interest.npy')[:,:,0]

                # Calculate cross correlation for each prediction image
                for j, pred_img in enumerate(pred_imgs):
                    correlation_values[i, j], mse_values[i, j] = correlation_and_mse(pred_img, ground_truth_img, mask)
                
            # Save correlation and mse values
            np.save(os.path.join(output_path, f'correlation_values_{max_var}_{sampling_step}.npy'), correlation_values)
            np.save(os.path.join(output_path, f'mse_values_{max_var}_{sampling_step}.npy'), mse_values)

            # Print values
            print(f'Max var: {max_var}, Sampling step: {sampling_step}')
            print(f'Correlation values: {correlation_values.mean(axis=0)}')
            print(f'MSE values: {mse_values.mean(axis=0)}')
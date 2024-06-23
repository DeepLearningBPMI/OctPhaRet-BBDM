import numpy as np

if __name__ == '__main__':
    sampling_steps = [20, 50, 100, 200]
    max_vars = {0.2: 'dataset_cropped_clipped_max_var_02',
                0.5: 'dataset_cropped_clipped_max_var_05',
                1: 'dataset_cropped_clipped'}

    output_path = '/scistor/guest/mhn744/BScProject/BBDM/evaluation_results'

    # print table header in LaTeX format
    # print rows 
    print('Max. var. ($s$) & Sampling step ($S$) & Avg. Correlation & Avg. MSE \\\\')
    for max_var, max_var_path in max_vars.items():
        for sampling_step in sampling_steps:
            correlation = np.load(f'correlation_values_{max_var}_{sampling_step}.npy')
            mse = np.load(f'mse_values_{max_var}_{sampling_step}.npy')

            print(f'{max_var} & {sampling_step} & {correlation.mean(axis=0).mean():.3f} & {mse.mean(axis=0).mean():.3f} \\\\')
    print('')

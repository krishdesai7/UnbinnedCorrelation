import numpy as np
import os
import concurrent.futures
import shutil

smearings = np.linspace(0, 0.75, 20)
fluctuate_data_size = True
n_bootstraps = 500

def generate_data(smearing):
    try:
        output_dir = f'input-samples/input-samples-smearing-{smearing:.4f}-v1b'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        data_size = 10**4
        sim_size = 10**5
        n_iterations = 5
        epsilon = 1e-8

        
        rng_seed = 5048
        rng = np.random.default_rng(seed = rng_seed )
        
        mu_true, var_true = 0.2, 0.81
        mu_gen, var_gen = 0.0, 1.0
        
        min_bin_width = 0.20
            
        truth = rng.normal(mu_true, np.sqrt(var_true), (data_size))
        data = rng.normal(truth, smearing)
        gen = rng.normal(mu_gen, np.sqrt(var_gen), (sim_size))
        sim = rng.normal(gen, smearing)
        
        bins = np.linspace(-3, 4, 15)
        # bins = [truth.min()]
        # i = 0
        # while bins[-1] < truth.max() and i < len(bins):
        #     for binhigh in np.linspace(bins[i] + epsilon, truth.max(), 200):
        #         in_bin = (truth > bins[i]) & (truth < binhigh)
        #         in_reco_bin = (data > bins[i]) & (data < binhigh)
        #         if np.sum(in_bin) > 0:
        #             purity = np.sum(in_bin & in_reco_bin) / np.sum(in_bin)
        #             if purity > (0.5):
        #                 binwid = binhigh - bins[-1]
        #                 if binwid < min_bin_width :
        #                     binhigh = bins[-1] + min_bin_width
        #                 i += 1
        #                 bins.append(binhigh)
        #                 break
        #     else:
        #         break
                
        # #-- add a last bin
        # bins.append(truth.max())
                
        # bins = np.array(bins[1:-1])
        bin_widths = np.diff(bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        n_bins = len(bins) - 1
        
        os.makedirs(output_dir, exist_ok=True)
        
        out_file = os.path.join(output_dir, 'config.txt')
        
        # List of parameters with their formatting
        params = [
            ('data_size', data_size, '%d'),
            ('sim_size', sim_size, '%d'),
            ('n_bootstraps', n_bootstraps, '%d'),
            ('rng_seed', rng_seed, '%d'),
            ('mu_true', mu_true, '%.4f'),
            ('mu_gen', mu_gen, '%.4f'),
            ('var_true', var_true, '%.4f'),
            ('var_gen', var_gen, '%.4f'),
            ('smearing', smearing, '%.2f'),
            ('min_bin_width', min_bin_width, '%.4f')
        ]
        
        # Write parameters to the config file
        with open(out_file, 'w') as text_file:
            for name, value, fmt in params:
                text_file.write(f'{name} {fmt % value}\n')
        
        binary_file_path = os.path.join(output_dir, 'binning.npy')
        text_file_path = os.path.join(output_dir, 'binning.txt')
    
        with open(binary_file_path, 'wb') as f:
            np.save(f, bins)
            np.save(f, bin_widths)
            np.save(f, bin_centers)
            np.save(f, n_bins)
        
        # Prepare data for the text file
        binning_data = {
            'bins': bins,
            'bin_widths': bin_widths,
            'bin_centers': bin_centers,
            'n_bins': n_bins
        }
        
        # Write data to the text file
        with open(text_file_path, 'w') as text_file:
            for name, array in binning_data.items():
                text_file.write(f'{name} {array}\n')
        
    
        for si in range( n_bootstraps ) :
            
            out_file = os.path.join(output_dir, f'sample-{si:04d}.npz')
            
        
        
            this_data_size = data_size
            if fluctuate_data_size:
                this_data_size = rng.poisson( data_size )
            
            truth = rng.normal(mu_true, np.sqrt(var_true), (this_data_size))
            data = rng.normal(truth, smearing)
            gen = rng.normal(mu_gen, np.sqrt(var_gen), (sim_size))
            sim = rng.normal(gen, smearing)    
            
            
            np.savez_compressed(out_file, truth=truth, data=data,
                            gen=gen, sim=sim)
            
        print(f'Saved sample to file {out_file}')
        return

    except Exception as e:
        print(f"An error occurred for smearing {smearing}: {e}")

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(generate_data, smearings)

if __name__ == '__main__':
    main()
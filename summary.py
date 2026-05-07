import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    os.system('cls')
    df = pd.read_csv(r'result\shrimp_detailed_results.csv')

    len_err = df['len_err'].values
    dm_err = df['girth_err'].values

    fig, ax = plt.subplots(ncols=2)
    x_min, x_max = -1, 1 
    tick_step = 0.1
    ticks = np.arange(x_min, x_max + tick_step, tick_step)

    ax[0].hist(len_err, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax[0].set_title('Length Error Distribution', fontweight='bold')
    ax[0].set_xticks(ticks) 
    ax[0].tick_params(axis='x', rotation=45) 
    ax[0].set_xlabel('Error (%)')
    ax[0].set_ylabel('Frequency')
    ax[0].grid(True, linestyle='--', alpha=0.6)

    ax[1].hist(dm_err, bins=100, color='salmon', edgecolor='black', alpha=0.7)
    ax[1].set_title('Diameter Error Distribution', fontweight='bold')
    ax[1].set_xticks(ticks) 
    ax[1].tick_params(axis='x', rotation=45) 
    ax[1].set_xlabel('Error (%)')
    ax[1].set_ylabel('Frequency')
    ax[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
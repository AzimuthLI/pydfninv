import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def exp_normal(x, mu, std):

    y = np.exp(-0.5*(x-mu)**2/std)

    return y#/(np.sqrt(2*np.pi)*std)

def log_normal(y):

    return np.log(y)

if __name__ == '__main__':

    df = pd.read_csv('/Volumes/SD_Card/Thesis_project/synthetic_model/output/obs_readings.csv', index_col=0).values / 1e6
    print(df)

    df += np.random.normal(0, 0.01, df.shape)
    print(df[:, 1])

    plt.plot(df[:, 0])
    plt.plot(df[:, 1])
    plt.plot(df[:, 2])
    plt.plot(df[:, 3])
    plt.plot(df[:, 4])
    plt.show()
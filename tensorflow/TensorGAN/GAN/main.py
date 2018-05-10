import net
import util
import config

import tensorflow as tf
import seaborn as sns
import numpy as np

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

def main():
    gan = net.GAN(config)
    gan.train(util.DataDistribution(), util.GeneratorDistribution(range=8), config)

if __name__=="__main__":
    main()
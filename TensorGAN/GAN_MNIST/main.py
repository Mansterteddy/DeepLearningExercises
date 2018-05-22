import tensorflow as tf
import numpy as np

import config
import net

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

def main():
    gan = net.GAN(config)
    gan.train()

if __name__ == "__main__":
    main()
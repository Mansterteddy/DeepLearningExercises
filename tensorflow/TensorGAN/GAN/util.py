import numpy as np
import matplotlib.pyplot as plt

class DataDistribution(object):
    def __init__(self):
        self.mu = 2
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range
    
    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

def samples(model, session, data, sample_range, batch_size, num_points=10000, num_bins=100):
    '''
    Return a tuple (db, pd, pg), where db is current decision boundary, pd is a histogram of samples from data distribution and pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    #decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i: batch_size * (i+1)] = session.run(model.D1, {model.x: np.reshape(xs[batch_size * i: batch_size * (i+1)], (batch_size, 1))})

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    #generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i: batch_size * (i+1)] = session.run(model.G, {model.z: np.reshape(zs[batch_size * i: batch_size * (i+1)], (batch_size, 1))})
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg

def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label="decision boundary")
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label="real data")
    plt.plot(p_x, pg, label="generated data")
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()

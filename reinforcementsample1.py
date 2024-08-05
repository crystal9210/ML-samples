import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import threading
import time
import gym
from gym import spaces

# Generate sample data
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# Define a simple RL environment for clustering
class ClusteringEnv(gym.Env):
    def __init__(self, X, n_clusters=3):
        super(ClusteringEnv, self).__init__()
        self.X = X
        self.n_clusters = n_clusters
        self.action_space = spaces.Discrete(n_clusters)
        self.observation_space = spaces.Box(low=np.min(X), high=np.max(X), shape=X.shape[1:])
        self.current_step = 0
        self.cluster_centers = np.zeros((n_clusters, X.shape[1]))

    def reset(self):
        self.current_step = 0
        self.cluster_centers = np.zeros((self.n_clusters, self.X.shape[1]))
        return self.X[self.current_step]

    def step(self, action):
        self.cluster_centers[action] += self.X[self.current_step]
        self.current_step += 1
        done = self.current_step >= len(self.X)
        return self.X[self.current_step] if not done else np.zeros_like(self.X[0]), 0, done, {}

def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

def plot_clusters(X, y_pred, title):
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title(title)
    plt.show()

# Define a simple RL agent for clustering
class SimpleRLAgent:
    def __init__(self, env):
        self.env = env

    def train(self, n_episodes=100):
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = np.random.choice(self.env.action_space.n)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state

    def predict(self):
        state = self.env.reset()
        predictions = []
        done = False
        while not done:
            action = np.random.choice(self.env.action_space.n)
            next_state, reward, done, _ = self.env.step(action)
            predictions.append(action)
            state = next_state
        return np.array(predictions)

# Create the RL environment
env = ClusteringEnv(X, n_clusters=3)

# Create and train the RL agent
rl_agent = SimpleRLAgent(env)
rl_agent.train(n_episodes=100)

# Predict clusters using the RL agent
rl_predictions = rl_agent.predict()

# Perform KMeans clustering for comparison
kmeans_labels, _ = kmeans_clustering(X, n_clusters=3)

# Plot the results
plot_clusters(X, rl_predictions, "RL Clustering")
plot_clusters(X, kmeans_labels, "KMeans Clustering")

# Compute accuracy for evaluation
def compute_accuracy(y_true, y_pred):
    y_true_clusters = [np.argmax(np.bincount(y_true[y_pred == i])) for i in range(len(np.unique(y_pred)))]
    y_true_mapped = np.array([y_true_clusters[label] for label in y_pred])
    return accuracy_score(y_true, y_true_mapped)

# Note: Since this is unsupervised learning, we don't have ground truth labels. We will use KMeans labels as a proxy.
accuracy_rl = compute_accuracy(kmeans_labels, rl_predictions)
accuracy_kmeans = compute_accuracy(kmeans_labels, kmeans_labels)

print(f"RL Clustering Accuracy: {accuracy_rl}")
print(f"KMeans Clustering Accuracy: {accuracy_kmeans}")

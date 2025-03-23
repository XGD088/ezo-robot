def test_means_k():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import torch
    import torch.nn as nn

    # Step 1: Create synthetic continuous input with "conceptual segments"
    np.random.seed(0)
    segment1 = np.ones(100) * 0.2  # flat region (state A)
    segment2 = np.sin(np.linspace(0, 3 * np.pi, 100))  # oscillating region (state B)
    segment3 = np.linspace(0, 1, 100)  # increasing region (state C)
    signal = np.concatenate([segment1, segment2, segment3])
    x_input = signal.reshape(-1, 1).astype(np.float32)

    # Step 2: Define a simple neural network to learn feature embeddings
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 2)  # output 2D embedding for visualization
            )

        def forward(self, x):
            return self.encoder(x)

    # Train the encoder (self-supervised way)
    model = SimpleEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x_tensor = torch.tensor(x_input)

    for _ in range(300):
        embedding = model(x_tensor)
        loss = torch.mean(embedding ** 2)  # dummy loss just to activate layers
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step 3: Extract embeddings
    with torch.no_grad():
        embeddings = model(x_tensor).numpy()

    # Step 4: Clustering in embedding space
    kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
    labels = kmeans.labels_

    # Step 5: Visualization
    plt.figure(figsize=(14, 5))

    # Plot original signal
    plt.subplot(1, 2, 1)
    plt.plot(signal, label="Input Signal")
    plt.title("Input Signal (Continuous)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # Plot clustered segments
    plt.subplot(1, 2, 2)
    plt.scatter(np.arange(len(signal)), signal, c=labels, cmap='viridis')
    plt.title("Auto-segmented 'Conceptual Paragraphs'")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


test_means_k()
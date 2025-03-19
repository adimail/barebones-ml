import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from neural_networks.rnn import RNN, create_sequences, mse, mse_grad
import matplotlib.pyplot as plt

df = pd.read_csv("data/clean_weather.csv")

df.ffill(inplace=True)
df.bfill(inplace=True)

predictors = ["tmax", "tmin", "rain"]
target = "tmax_tomorrow"
X = df[predictors].values
y = df[target].values

n = len(df)
train_size = int(0.7 * n)
valid_size = int(0.15 * n)
test_size = n - train_size - valid_size

train_x, valid_x, test_x = np.split(X, [train_size, train_size + valid_size])
train_y, valid_y, test_y = np.split(y, [train_size, train_size + valid_size])

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
valid_x_scaled = scaler.transform(valid_x)
test_x_scaled = scaler.transform(test_x)

seq_len = 10
train_sequences, train_targets = create_sequences(train_x_scaled, train_y, seq_len)
valid_sequences, valid_targets = create_sequences(valid_x_scaled, valid_y, seq_len)
test_sequences, test_targets = create_sequences(test_x_scaled, test_y, seq_len)

input_size = len(predictors)  # 3 features: tmax, tmin, rain
hidden_size = 5
output_size = 1
rnn = RNN(input_size, hidden_size, output_size)

learning_rate = 1e-6
num_epochs = 30

train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    indices = np.arange(len(train_sequences))
    np.random.shuffle(indices)
    train_sequences_shuffled = [train_sequences[i] for i in indices]
    train_targets_shuffled = [train_targets[i] for i in indices]

    total_loss = 0
    for seq, target in zip(train_sequences_shuffled, train_targets_shuffled):
        outputs, hiddens = rnn.forward(seq)
        d_outputs = mse_grad(target.reshape(-1, 1), outputs)
        loss = mse(target.reshape(-1, 1), outputs)
        total_loss += loss
        gradients = rnn.backward(d_outputs, hiddens, seq)
        rnn.update(gradients, learning_rate)

    avg_train_loss = total_loss / len(train_sequences)
    train_losses.append(avg_train_loss)

    valid_loss = 0
    for seq, target in zip(valid_sequences, valid_targets):
        outputs, _ = rnn.forward(seq)
        loss = mse(target.reshape(-1, 1), outputs)
        valid_loss += loss
    avg_valid_loss = valid_loss / len(valid_sequences)
    valid_losses.append(avg_valid_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}"
    )

test_loss = 0
for seq, target in zip(test_sequences, test_targets):
    outputs, _ = rnn.forward(seq)
    loss = mse(target.reshape(-1, 1), outputs)
    test_loss += loss
print(f"Test Loss: {test_loss / len(test_sequences):.6f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

sample_seq = test_sequences[0]
sample_target = test_targets[0]
outputs, _ = rnn.forward(sample_seq)
predicted = outputs.flatten()
actual = sample_target
plt.figure(figsize=(12, 6))
plt.plot(range(len(predicted)), predicted, label="Predicted", marker="o")
plt.plot(range(len(actual)), actual, label="Actual", marker="x")
plt.xlabel("Time Step")
plt.ylabel("tmax_tomorrow")
plt.title("Predicted vs Actual tmax_tomorrow for a Sample Sequence")
plt.legend()
plt.grid(True)
plt.show()

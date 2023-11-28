import sys
import pandas as pd
import matplotlib.pyplot as plt

colors = ['green', 'blue', 'red', 'black']

# Check if a file path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

file_paths = sys.argv[1:]

window = 100000

# Load the data
dfs = []

min_len = float('inf')
for fp in file_paths:
    dfs.append(pd.read_csv(fp))
    min_len = min(min_len, len(dfs[-1]))

# dfs = [df.head(min_len) for df in dfs]

# Creating subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Plotting moving average of the first column
for i, df in enumerate(dfs):
    moving_avgp = df['prob_word'].rolling(window=window).mean()
    ax1.plot(moving_avgp, color=colors[i], label=df['model'][0])

ax1.set_title('{}-Instance Moving Average of P(word|context)'.format(window))
ax1.set_ylabel('P(word|context)')

# Plotting moving average of the second column
for i, df in enumerate(dfs):
    moving_avga = df['correct'].rolling(window=window).mean()
    ax2.plot(moving_avga, color=colors[i], label=df['model'][0])
ax2.set_title('{}-Instance Moving Average of Accuracy'.format(window))
ax2.set_ylabel('Accuracy')

# ax3.set_title('Vocab Size')
# ax3.plot(df['vocab_size'])
# ax3.set_xlabel('# training instances')
# ax3.set_ylabel('Vocab Size')

plt.legend()
plt.tight_layout()
plt.show()


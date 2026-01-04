import matplotlib.pyplot as plt
import re

def parse_log_file(filename):
    data = []
    # Parse k, epoch, and val_loss
    pattern = re.compile(r"k=\s*(\d+)\s+e=\s*(\d+).*?val_loss=([\d\.]+)")
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    data.append({
                        'k': int(match.group(1)),
                        'epoch': int(match.group(2)),
                        'val_loss': float(match.group(3))
                    })
    except FileNotFoundError:
        print(f"Warning: Could not find {filename}")
    return data

# --- 1. Load Data ---
part1 = parse_log_file('info_part1.txt')
part2 = parse_log_file('info_part2.txt')

# Combine the lists (No need to fix time, since k is consistent)
full_run = part1 + part2

if not full_run:
    print("No data found! Make sure log_part1.txt and log_part2.txt exist.")
    exit()

# --- 2. Calculate X-Axis (Continuous K) ---
# We add (epoch / 70) to k. 
# Since max epoch is 60, this creates a value like 5.0, 5.1... 5.8 
# This separates the stages visually with a small gap.
k_continuous = [x['k'] + (x['epoch'] / 70.0) for x in full_run]
val_losses = [x['val_loss'] for x in full_run]

# --- 3. Plotting ---
plt.figure(figsize=(12, 6))

# Plot the main line
plt.plot(k_continuous, val_losses, label='Validation Loss', color='#1f77b4', linewidth=1.5)

# Highlight the "Breakthrough" point (k=20)
#plt.axvline(x=20, color='green', linestyle='--', alpha=0.8, label='Breakthrough (k=20)')

# Add formatting
plt.title(f"Training Progress by Curriculum Stage (k)\nCurrent Best Loss: {min(val_losses):.6f}", fontsize=14)
plt.xlabel("Curriculum Stage (k)", fontsize=12)
plt.ylabel("Validation Loss", fontsize=12)

# Set X-ticks to be whole integers (0, 2, 4...) so it's easy to read
plt.xticks(range(0, 30, 2))
plt.grid(True, alpha=0.3)
plt.legend()

# Save and Show
plt.tight_layout()
plt.savefig("training_by_k.png")
print("Plot saved as 'training_by_k.png'")
plt.show()

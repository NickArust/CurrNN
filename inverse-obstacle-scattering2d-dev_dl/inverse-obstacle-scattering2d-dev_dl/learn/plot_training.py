import matplotlib.pyplot as plt
import re

def parse_log_file(filename):
    data = []
    # Regex to extract k, epoch, val_loss, and time
    # Matches lines like: INFO:root:k= 0 e=  10 ... val_loss=0.001397 ... time=4238.8s
    pattern = re.compile(r"k=\s*(\d+)\s+e=\s*(\d+).*?val_loss=([\d\.]+).*?time=([\d\.]+)s")
    
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                data.append({
                    'k': int(match.group(1)),
                    'epoch': int(match.group(2)),
                    'val_loss': float(match.group(3)),
                    'time': float(match.group(4))
                })
    return data

# --- 1. Load Data ---
part1 = parse_log_file('info_part1.txt')
part2 = parse_log_file('info_part2.txt')

# --- 2. Stitch Data (Handle Time Reset) ---
# We need to offset Part 2's time by the final timestamp of Part 1
time_offset = part1[-1]['time'] if part1 else 0

full_run = part1.copy()
for entry in part2:
    entry['time'] += time_offset  # Add the offset to make time continuous
    full_run.append(entry)

# --- 3. Prepare Plotting Arrays ---
times = [x['time'] / 3600 for x in full_run]  # Convert seconds to hours
val_losses = [x['val_loss'] for x in full_run]
k_stages = [x['k'] for x in full_run]

# --- 4. Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(times, val_losses, label='Validation Loss', color='#1f77b4', linewidth=2)

# Highlight the "Breakthrough" point (Start of k=20)
breakthrough_index = next((i for i, x in enumerate(full_run) if x['k'] == 20), None)
if breakthrough_index:
    plt.axvline(x=times[breakthrough_index], color='green', linestyle='--', alpha=0.7, label='Breakthrough (k=20)')

# Annotate the Phases
plt.text(times[5], max(val_losses)*0.9, 'Phase 1:\nInitial Descent', color='black', fontsize=10)
plt.text(times[len(part1)-10], 0.00018, 'Phase 2:\nThe Plateau', color='red', fontsize=10, ha='center')
plt.text(times[-1], val_losses[-1] + 0.00005, 'Phase 3:\nBreakthrough', color='green', fontsize=10, ha='right')

# Formatting
plt.title(f"Full Training Run Analysis (Combined Parts 1 & 2)\nCurrent Best Loss: {min(val_losses):.6f}", fontsize=14)
plt.xlabel("Training Time (Hours)", fontsize=12)
plt.ylabel("Validation Loss", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Save and Show
plt.tight_layout()
plt.savefig("training_breakthrough.png")
plt.show()

print(f"Plot saved as 'training_breakthrough.png'.")
print(f"Total stitched training time: {times[-1]:.1f} hours.")

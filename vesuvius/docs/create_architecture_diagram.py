import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Left subplot: Pooling Decision Algorithm
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.axis('off')
ax1.set_title('Pooling Operation Decision Algorithm', fontsize=16, fontweight='bold', pad=20)

# Define box style
box_style = "round,pad=0.3"

# Start box
start_box = FancyBboxPatch((3.5, 12.5), 3, 0.8, boxstyle=box_style, 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
ax1.add_patch(start_box)
ax1.text(5, 12.9, 'Start', ha='center', va='center', fontsize=12, fontweight='bold')

# Input parameters
input_box = FancyBboxPatch((2, 11), 6, 1.2, boxstyle=box_style,
                          facecolor='lightblue', edgecolor='black', linewidth=2)
ax1.add_patch(input_box)
ax1.text(5, 11.6, 'Input: spacing, patch_size,\nmin_feature_map_size, max_numpool', 
         ha='center', va='center', fontsize=10)

# Initialize variables
init_box = FancyBboxPatch((2.5, 9.3), 5, 1.2, boxstyle=box_style,
                         facecolor='lightyellow', edgecolor='black', linewidth=2)
ax1.add_patch(init_box)
ax1.text(5, 9.9, 'Initialize:\ncurrent_spacing = spacing\ncurrent_size = patch_size', 
         ha='center', va='center', fontsize=9)

# Main loop
loop_box = FancyBboxPatch((3, 7.8), 4, 0.8, boxstyle=box_style,
                         facecolor='lightcoral', edgecolor='black', linewidth=2)
ax1.add_patch(loop_box)
ax1.text(5, 8.2, 'While True:', ha='center', va='center', fontsize=11, fontweight='bold')

# Check 1: Size constraint
check1_box = FancyBboxPatch((1, 6.5), 8, 0.8, boxstyle=box_style,
                           facecolor='wheat', edgecolor='black', linewidth=1.5)
ax1.add_patch(check1_box)
ax1.text(5, 6.9, 'Check 1: current_size[i] ≥ 2 × min_feature_map_size', 
         ha='center', va='center', fontsize=9)

# Check 2: Spacing ratio
check2_box = FancyBboxPatch((1, 5.3), 8, 0.8, boxstyle=box_style,
                           facecolor='wheat', edgecolor='black', linewidth=1.5)
ax1.add_patch(check2_box)
ax1.text(5, 5.7, 'Check 2: current_spacing[i] / min_spacing < 2', 
         ha='center', va='center', fontsize=9)

# Check 3: Max pooling operations
check3_box = FancyBboxPatch((1, 4.1), 8, 0.8, boxstyle=box_style,
                           facecolor='wheat', edgecolor='black', linewidth=1.5)
ax1.add_patch(check3_box)
ax1.text(5, 4.5, 'Check 3: num_pool_per_axis[i] < max_numpool', 
         ha='center', va='center', fontsize=9)

# Decision diamond
decision_points = [(5, 3.2), (6.5, 2.2), (5, 1.2), (3.5, 2.2)]
decision_diamond = patches.Polygon(decision_points, facecolor='lightsteelblue', 
                                  edgecolor='black', linewidth=2)
ax1.add_patch(decision_diamond)
ax1.text(5, 2.2, 'Valid axes\navailable?', ha='center', va='center', fontsize=9)

# No valid axes - break
break_box = FancyBboxPatch((7, 1.8), 2, 0.8, boxstyle=box_style,
                          facecolor='salmon', edgecolor='black', linewidth=2)
ax1.add_patch(break_box)
ax1.text(8, 2.2, 'Break', ha='center', va='center', fontsize=10, fontweight='bold')

# Yes - update parameters
update_box = FancyBboxPatch((0.5, 0.3), 4, 1.2, boxstyle=box_style,
                           facecolor='lightgreen', edgecolor='black', linewidth=1.5)
ax1.add_patch(update_box)
ax1.text(2.5, 0.9, 'Update:\n• pool_kernel_sizes\n• current_spacing × 2\n• current_size ÷ 2', 
         ha='center', va='center', fontsize=8)

# Arrows
arrow_props = dict(arrowstyle='->', lw=2, color='black')
# Flow arrows
ax1.annotate('', xy=(5, 11), xytext=(5, 12.5), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 9.3), xytext=(5, 11), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 7.8), xytext=(5, 9.3), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 6.5), xytext=(5, 7.8), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 5.3), xytext=(5, 6.5), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 4.1), xytext=(5, 5.3), arrowprops=arrow_props)
ax1.annotate('', xy=(5, 3.2), xytext=(5, 4.1), arrowprops=arrow_props)
# Decision arrows
ax1.annotate('No', xy=(7, 2.2), xytext=(6.5, 2.2), arrowprops=arrow_props)
ax1.annotate('Yes', xy=(3.5, 2.2), xytext=(2.5, 1.5), arrowprops=arrow_props, ha='right')
# Loop back arrow
ax1.annotate('', xy=(0.5, 8.2), xytext=(0.5, 0.9), 
             arrowprops=dict(arrowstyle='->', lw=2, color='blue', connectionstyle="arc3,rad=.3"))

# Right subplot: Stage Configuration
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.axis('off')
ax2.set_title('Network Stage Configuration', fontsize=16, fontweight='bold', pad=20)

# Stage determination
stage_header = FancyBboxPatch((2, 12), 6, 1, boxstyle=box_style,
                             facecolor='lightblue', edgecolor='black', linewidth=2)
ax2.add_patch(stage_header)
ax2.text(5, 12.5, 'Stage Determination', ha='center', va='center', fontsize=12, fontweight='bold')

# Number of stages formula
formula_box = FancyBboxPatch((1.5, 10.5), 7, 0.8, boxstyle=box_style,
                            facecolor='lightyellow', edgecolor='black', linewidth=1.5)
ax2.add_patch(formula_box)
ax2.text(5, 10.9, 'num_stages = len(pool_op_kernel_sizes)', ha='center', va='center', fontsize=10)

# Blocks per stage
blocks_header = FancyBboxPatch((2, 9), 6, 0.8, boxstyle=box_style,
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
ax2.add_patch(blocks_header)
ax2.text(5, 9.4, 'Blocks per Stage', ha='center', va='center', fontsize=11, fontweight='bold')

# Stage-specific block counts
stage_configs = [
    ("Stage 0", "1 block", 7.8),
    ("Stage 1", "3 blocks", 6.8),
    ("Stage 2", "4 blocks", 5.8),
    ("Stage 3+", "6 blocks", 4.8)
]

for stage, blocks, y_pos in stage_configs:
    stage_box = FancyBboxPatch((1, y_pos), 3.5, 0.7, boxstyle=box_style,
                              facecolor='wheat', edgecolor='black', linewidth=1)
    ax2.add_patch(stage_box)
    ax2.text(2.75, y_pos + 0.35, stage, ha='center', va='center', fontsize=9, fontweight='bold')
    
    blocks_box = FancyBboxPatch((5.5, y_pos), 3, 0.7, boxstyle=box_style,
                               facecolor='lightcoral', edgecolor='black', linewidth=1)
    ax2.add_patch(blocks_box)
    ax2.text(7, y_pos + 0.35, blocks, ha='center', va='center', fontsize=9)

# Features per stage
features_header = FancyBboxPatch((2, 3.5), 6, 0.8, boxstyle=box_style,
                                facecolor='lightgreen', edgecolor='black', linewidth=2)
ax2.add_patch(features_header)
ax2.text(5, 3.9, 'Features per Stage', ha='center', va='center', fontsize=11, fontweight='bold')

# Feature calculation
feature_box = FancyBboxPatch((1, 2.2), 8, 0.8, boxstyle=box_style,
                            facecolor='lightyellow', edgecolor='black', linewidth=1.5)
ax2.add_patch(feature_box)
ax2.text(5, 2.6, 'features[i] = min(base_features × 2^i, max_features)', 
         ha='center', va='center', fontsize=9)

# Default values
defaults_box = FancyBboxPatch((1.5, 0.8), 7, 0.8, boxstyle=box_style,
                             facecolor='lightsteelblue', edgecolor='black', linewidth=1.5)
ax2.add_patch(defaults_box)
ax2.text(5, 1.2, 'base_features = 32, max_features = 320', 
         ha='center', va='center', fontsize=9)

# Add example calculation
example_text = """Example for 3D input:
• Patch: [96, 96, 96], Spacing: [1, 1, 1]
• Stage 0: 32 features, kernel [3,3,3]
• Stage 1: 64 features, pool [2,2,2]
• Stage 2: 128 features, pool [2,2,2]
• ... continues until size constraints"""

ax2.text(5, 0.2, example_text, ha='center', va='top', fontsize=8, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lavender', alpha=0.5))

plt.tight_layout()
plt.savefig('architecture_decision_flow.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second diagram showing the adaptive channel behavior
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Adaptive Channel Configuration', fontsize=16, fontweight='bold', pad=20)

# Input section
input_header = FancyBboxPatch((1, 8), 10, 1, boxstyle=box_style,
                             facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(input_header)
ax.text(6, 8.5, 'Input Channel Detection', ha='center', va='center', fontsize=12, fontweight='bold')

# Sources of input channels
sources = [
    "1. ConfigManager.in_channels",
    "2. Dataset channel detection", 
    "3. Default: 1 channel"
]

y_start = 6.8
for i, source in enumerate(sources):
    source_box = FancyBboxPatch((2, y_start - i*0.8), 8, 0.6, boxstyle=box_style,
                               facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(source_box)
    ax.text(6, y_start - i*0.8 + 0.3, source, ha='center', va='center', fontsize=9)

# Output configuration
output_header = FancyBboxPatch((1, 3.5), 10, 1, boxstyle=box_style,
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
ax.add_patch(output_header)
ax.text(6, 4, 'Output Channel Configuration (Per Task)', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Decision flow for output channels
decision_y = 2.5
decision_points = [(6, decision_y + 0.5), (7.5, decision_y), (6, decision_y - 0.5), (4.5, decision_y)]
decision_diamond = patches.Polygon(decision_points, facecolor='lightsteelblue', 
                                  edgecolor='black', linewidth=2)
ax.add_patch(decision_diamond)
ax.text(6, decision_y, 'Task specifies\nchannels?', ha='center', va='center', fontsize=8)

# Yes path
yes_box = FancyBboxPatch((8, decision_y - 0.3), 3, 0.6, boxstyle=box_style,
                        facecolor='lightcoral', edgecolor='black', linewidth=1)
ax.add_patch(yes_box)
ax.text(9.5, decision_y, 'Use specified', ha='center', va='center', fontsize=9)

# No path  
no_box = FancyBboxPatch((1, decision_y - 0.3), 3, 0.6, boxstyle=box_style,
                       facecolor='lightgreen', edgecolor='black', linewidth=1)
ax.add_patch(no_box)
ax.text(2.5, decision_y, 'Match input channels', ha='center', va='center', fontsize=9)

# Arrows
ax.annotate('Yes', xy=(8, decision_y), xytext=(7.5, decision_y), arrowprops=arrow_props)
ax.annotate('No', xy=(4, decision_y), xytext=(4.5, decision_y), arrowprops=arrow_props)

# Example configuration
example_box = FancyBboxPatch((1, 0.2), 10, 1.3, boxstyle=box_style,
                            facecolor='lavender', edgecolor='black', linewidth=1.5)
ax.add_patch(example_box)
example_text = """Example: 3-channel input
• adaptive_task: {} → outputs 3 channels (matches input)
• fixed_task: {"channels": 1} → outputs 1 channel (specified)
• multi_task: {"channels": 5} → outputs 5 channels (specified)"""
ax.text(6, 0.85, example_text, ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('adaptive_channel_flow.png', dpi=300, bbox_inches='tight')
plt.close()

print("Architecture diagrams created successfully!")
print("- architecture_decision_flow.png")
print("- adaptive_channel_flow.png")

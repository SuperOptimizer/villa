import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Create figure for high-level training flow
def create_training_flow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Define box style
    def create_box(x, y, w, h, text, color='lightblue', style='round'):
        if style == 'round':
            box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1', 
                                facecolor=color, edgecolor='black', linewidth=2)
        else:
            box = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold', wrap=True)

    # Define arrow
    def create_arrow(x1, y1, x2, y2, text=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, text, fontsize=9, ha='center')

    # Create main flow diagram
    create_box(3.5, 18, 3, 1, 'User runs train.py', 'lightgreen')
    create_arrow(5, 18, 5, 17.2)

    create_box(3.5, 16.5, 3, 0.7, 'Parse CLI Arguments', 'lightblue')
    create_arrow(5, 16.5, 5, 15.7)

    create_box(3.5, 15, 3, 0.7, 'Initialize ConfigManager', 'lightblue')
    create_arrow(5, 15, 5, 14.2)

    # Config branch
    create_box(2, 13.5, 2.5, 0.7, 'Load YAML Config', 'lightyellow')
    create_box(5.5, 13.5, 2.5, 0.7, 'Use Default Config', 'lightyellow')
    create_arrow(4, 14.2, 3.25, 13.5 + 0.7, 'Config File?')
    create_arrow(6, 14.2, 6.75, 13.5 + 0.7, 'No Config')
    create_arrow(3.25, 13.5, 5, 12.7)
    create_arrow(6.75, 13.5, 5, 12.7)

    create_box(3.5, 12, 3, 0.7, 'Update Config from CLI', 'lightblue')
    create_arrow(5, 12, 5, 11.2)

    create_box(3, 10.5, 4, 0.7, 'Auto-detect Targets from Data', 'lightblue')
    create_arrow(5, 10.5, 5, 9.7)

    create_box(3, 9, 4, 0.7, 'Build NetworkFromConfig', 'lightcoral')
    create_arrow(5, 9, 5, 8.2)

    # Model branch
    create_box(1.5, 7.5, 3, 0.7, 'Auto-configure Model', 'lightyellow')
    create_box(5.5, 7.5, 3, 0.7, 'Use Manual Config', 'lightyellow')
    create_arrow(3.5, 8.2, 3, 7.5 + 0.7, 'Auto?')
    create_arrow(6.5, 8.2, 7, 7.5 + 0.7, 'Manual')
    create_arrow(3, 7.5, 5, 6.7)
    create_arrow(7, 7.5, 5, 6.7)

    create_box(3, 6, 4, 0.7, 'Initialize Dataset', 'lightgreen')
    create_arrow(5, 6, 5, 5.2)

    # Dataset branch
    create_box(0.5, 4.5, 2, 0.7, 'ZarrDataset', 'lightyellow')
    create_box(3.5, 4.5, 2, 0.7, 'TifDataset', 'lightyellow')
    create_box(6.5, 4.5, 2, 0.7, 'NapariDataset', 'lightyellow')
    create_arrow(3, 5.2, 1.5, 4.5 + 0.7, 'zarr')
    create_arrow(5, 5.2, 4.5, 4.5 + 0.7, 'tif')
    create_arrow(7, 5.2, 7.5, 4.5 + 0.7, 'napari')
    create_arrow(1.5, 4.5, 5, 3.7)
    create_arrow(4.5, 4.5, 5, 3.7)
    create_arrow(7.5, 4.5, 5, 3.7)

    create_box(3, 3, 4, 0.7, 'Find Valid Patches', 'lightblue')
    create_arrow(5, 3, 5, 2.2)

    create_box(3, 2.5, 4, 0.7, 'Create DataLoaders', 'lightblue')
    create_arrow(5, 2.5, 5, 1.7)

    create_box(3, 1, 4, 0.7, 'Training Loop', 'lightcoral')
    create_arrow(7, 1.35, 8, 1.35)
    create_box(8, 1, 1.5, 0.7, 'Validate', 'lightyellow')
    create_arrow(8.75, 1, 8.75, 0.5)
    create_arrow(8.75, 0.5, 2.5, 0.5)
    create_arrow(2.5, 0.5, 2.5, 1.35)
    create_arrow(2.5, 1.35, 3, 1.35, 'Next Epoch')

    plt.title('Training Pipeline Flow', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('training_flow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created training flow diagram')


def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define box style
    def create_box(x, y, w, h, text, color='lightblue'):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, weight='bold')
    
    # Define arrow
    def create_arrow(x1, y1, x2, y2, style='->', color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, lw=2, color=color))
    
    # Input
    create_box(0.5, 4.5, 1.5, 1, 'Input\nB×C×D×H×W', 'lightgreen')
    
    # Encoder stages
    encoder_x = 2.5
    encoder_stages = [
        ('Stem\n32 ch', 8.5),
        ('Stage 1\n64 ch', 7),
        ('Stage 2\n128 ch', 5.5),
        ('Stage 3\n256 ch', 4),
        ('Stage 4\n320 ch', 2.5),
        ('Bottleneck\n320 ch', 1)
    ]
    
    skip_positions = []
    for i, (stage_text, y_pos) in enumerate(encoder_stages):
        create_box(encoder_x, y_pos, 2, 0.8, stage_text, 'lightblue')
        if i < len(encoder_stages) - 1:
            create_arrow(encoder_x + 1, y_pos, encoder_x + 1, y_pos - 0.7)
            if i < 5:  # Skip connections
                skip_positions.append((encoder_x + 2, y_pos + 0.4))
    
    # Connect input to encoder
    create_arrow(2, 5, 2.5, 8.5 + 0.4)
    
    # Task 1 Decoder
    decoder1_x = 6.5
    decoder1_stages = [
        ('Dec1 Stage 4', 2.5),
        ('Dec1 Stage 3', 4),
        ('Dec1 Stage 2', 5.5),
        ('Dec1 Stage 1', 7),
        ('Output Head\nTask: ink', 8.5)
    ]
    
    for i, (stage_text, y_pos) in enumerate(decoder1_stages):
        create_box(decoder1_x, y_pos, 2, 0.8, stage_text, 'lightcoral')
        if i < len(decoder1_stages) - 1:
            create_arrow(decoder1_x + 1, y_pos + 0.8, decoder1_x + 1, y_pos + 1.3)
    
    # Task 2 Decoder
    decoder2_x = 10
    decoder2_stages = [
        ('Dec2 Stage 4', 2.5),
        ('Dec2 Stage 3', 4),
        ('Dec2 Stage 2', 5.5),
        ('Dec2 Stage 1', 7),
        ('Output Head\nTask: damage', 8.5)
    ]
    
    for i, (stage_text, y_pos) in enumerate(decoder2_stages):
        create_box(decoder2_x, y_pos, 2, 0.8, stage_text, 'lightsalmon')
        if i < len(decoder2_stages) - 1:
            create_arrow(decoder2_x + 1, y_pos + 0.8, decoder2_x + 1, y_pos + 1.3)
    
    # Connect bottleneck to decoders
    create_arrow(4.5, 1.4, 6.5, 2.9)
    create_arrow(4.5, 1.4, 10, 2.9)
    
    # Skip connections
    skip_decoder_positions = [(2.5, 4, 5.5), (2.5, 5.5, 7), (2.5, 7, 8.5)]
    for i, (enc_y, dec1_y, dec2_y) in enumerate(skip_decoder_positions):
        # To decoder 1
        create_arrow(4.5, enc_y + 0.4, 6.5, dec1_y + 0.4, color='blue', style='->')
        # To decoder 2
        create_arrow(4.5, enc_y + 0.4, 10, dec2_y + 0.4, color='blue', style='->')
    
    # Outputs
    create_box(6.5, 9.5, 2, 0.5, 'Sigmoid', 'yellow')
    create_box(10, 9.5, 2, 0.5, 'Sigmoid', 'yellow')
    create_arrow(7.5, 9.3, 7.5, 9.5)
    create_arrow(11, 9.3, 11, 9.5)
    
    plt.title('Multi-Task U-Net Architecture', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created architecture diagram')


def create_dataset_flow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    def create_box(x, y, w, h, text, color='lightblue'):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', 
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)
    
    def create_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Data directory structure
    create_box(0.5, 6, 3, 1.5, 'Data Directory\n├── images/\n├── labels/\n└── masks/', 'lightgreen')
    
    # Load volumes
    create_box(4.5, 6.5, 2.5, 0.8, 'Load Volumes', 'lightblue')
    create_arrow(3.5, 6.75, 4.5, 6.9)
    
    # Group by target
    create_box(7.5, 6.5, 2.5, 0.8, 'Group by Target', 'lightblue')
    create_arrow(7, 6.9, 7.5, 6.9)
    
    # Find patches
    create_box(5, 4.5, 3, 0.8, 'Find Valid Patches', 'lightcoral')
    create_arrow(8.75, 6.5, 6.5, 5.3)
    
    # Check mask
    create_box(3, 3, 2, 0.8, 'Use Mask?', 'lightyellow')
    create_box(7, 3, 2, 0.8, 'Use Label > 0', 'lightyellow')
    create_arrow(6.5, 4.5, 4, 3.8)
    create_arrow(6.5, 4.5, 8, 3.8)
    
    # Filter
    create_box(4.5, 1.5, 3, 0.8, 'Filter by min_ratio', 'lightblue')
    create_arrow(4, 3, 6, 2.3)
    create_arrow(8, 3, 6, 2.3)
    
    # Store
    create_box(8.5, 1.5, 3, 0.8, 'Store Coordinates', 'lightgreen')
    create_arrow(7.5, 1.9, 8.5, 1.9)
    
    # Data loading
    create_box(0.5, 0.2, 2, 0.8, 'Get Patch', 'lightcoral')
    create_box(3, 0.2, 2, 0.8, 'Normalize', 'lightblue')
    create_box(5.5, 0.2, 2, 0.8, 'Augment', 'lightblue')
    create_box(8, 0.2, 2, 0.8, 'To Tensor', 'lightgreen')
    
    create_arrow(2.5, 0.6, 3, 0.6)
    create_arrow(5, 0.6, 5.5, 0.6)
    create_arrow(7.5, 0.6, 8, 0.6)
    
    plt.title('Dataset Pipeline Flow', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('dataset_flow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created dataset flow diagram')


def create_training_loop_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    def create_box(x, y, w, h, text, color='lightblue'):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', 
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)
    
    def create_arrow(x1, y1, x2, y2, text=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, text, fontsize=8, ha='center')
    
    # Training epoch
    create_box(3.5, 10.5, 3, 0.8, 'Start Epoch', 'lightgreen')
    create_arrow(5, 10.5, 5, 9.7)
    
    create_box(3.5, 9, 3, 0.7, 'model.train()', 'lightblue')
    create_arrow(5, 9, 5, 8.2)
    
    # Batch loop
    create_box(3.5, 7.5, 3, 0.7, 'Get Batch', 'lightblue')
    create_arrow(5, 7.5, 5, 6.7)
    
    create_box(3.5, 6, 3, 0.7, 'Forward Pass', 'lightcoral')
    create_arrow(5, 6, 5, 5.2)
    
    # Multi-task loss
    create_box(1, 4.5, 2, 0.7, 'Task 1 Loss', 'lightyellow')
    create_box(4, 4.5, 2, 0.7, 'Task 2 Loss', 'lightyellow')
    create_box(7, 4.5, 2, 0.7, 'Task N Loss', 'lightyellow')
    
    create_arrow(3, 5.2, 2, 5.2)
    create_arrow(5, 5.2, 5, 5.2)
    create_arrow(7, 5.2, 8, 5.2)
    
    create_box(3.5, 3.3, 3, 0.7, 'Sum Weighted', 'lightblue')
    create_arrow(2, 4.5, 5, 4)
    create_arrow(5, 4.5, 5, 4)
    create_arrow(8, 4.5, 5, 4)
    
    create_arrow(5, 3.3, 5, 2.5)
    
    # Backward
    create_box(3.5, 1.8, 3, 0.7, 'Backward Pass', 'lightcoral')
    create_arrow(5, 1.8, 5, 1)
    
    create_box(3.5, 0.3, 3, 0.7, 'Optimizer Step', 'lightblue')
    
    # Loop back
    create_arrow(3.5, 0.65, 1, 0.65)
    create_arrow(1, 0.65, 1, 7.85)
    create_arrow(1, 7.85, 3.5, 7.85, 'Next Batch')
    
    # Validation branch
    create_arrow(6.5, 7.85, 8.5, 7.85)
    create_box(8.5, 7.5, 1.5, 0.7, 'Validate', 'lightgreen')
    
    plt.title('Training Loop Flow', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('training_loop_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created training loop diagram')


if __name__ == '__main__':
    create_training_flow_diagram()
    create_architecture_diagram()
    create_dataset_flow_diagram()
    create_training_loop_diagram()
    print('All diagrams created successfully!')

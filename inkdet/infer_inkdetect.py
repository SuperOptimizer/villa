import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import zarr
import cv2
from tqdm import tqdm

# Import your model classes (from the training script)
from train_inkdetect import InkDetectionModel, CFG


class InferenceDataset(Dataset):
    """Simple dataset for inference chunks"""

    def __init__(self, chunks, xyxys):
        self.chunks = chunks
        self.xyxys = xyxys

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        xy = self.xyxys[idx]

        # Normalize
        chunk = np.clip(chunk, 0, 200) / 255.0
        chunk_tensor = torch.from_numpy(chunk).float()

        return chunk_tensor, xy


def get_fragment_chunks(zarr_path, fragment_id, masks_path):
    """Extract valid chunks from a fragment"""
    zarr_store = zarr.open(zarr_path, mode='r')
    frag_data = zarr_store[fragment_id]
    d, h, w = frag_data.shape

    # Load fragment mask
    frag_mask_path = os.path.join(masks_path, fragment_id, f"{fragment_id}_mask.png")
    if not os.path.exists(frag_mask_path):
        print(f"No fragment mask found for {fragment_id}")
        return None, None, None

    frag_mask = cv2.imread(frag_mask_path, 0)
    frag_mask = cv2.resize(frag_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Collect valid chunks
    chunks = []
    xyxys = []

    for y in range(0, h - CFG.size, CFG.stride):
        for x in range(0, w - CFG.size, CFG.stride):
            # Check if chunk is valid (inside fragment mask)
            chunk_mask = frag_mask[y:y + CFG.size, x:x + CFG.size]
            if np.all(chunk_mask > 0):
                # Extract 3D chunk
                chunk_3d = frag_data[:, y:y + CFG.size, x:x + CFG.size].astype(np.float32)

                # Apply voxel threshold
                chunk_3d[chunk_3d < CFG.voxel_threshold] = 0

                chunks.append(chunk_3d)
                xyxys.append([x, y, x + CFG.size, y + CFG.size])

    return chunks, xyxys, (h, w)


@torch.no_grad()
def run_inference(model, dataloader, output_shape, device):
    """Run inference and assemble 2D output"""
    # Initialize output arrays
    # Since we predict 4x4x4 and want 2D, we'll average over Z and use 4x4 spatial
    scale_factor = CFG.size // CFG.output_size  # 64/4 = 16

    h_out = output_shape[0] // scale_factor
    w_out = output_shape[1] // scale_factor

    pred_sum = np.zeros((h_out, w_out), dtype=np.float32)
    pred_count = np.zeros((h_out, w_out), dtype=np.float32)

    model.eval()

    for chunks, xyxys in tqdm(dataloader, desc="Running inference"):
        chunks = chunks.to(device)

        # Get predictions
        outputs = model(chunks)  # (B, 1, 4, 4, 4)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)

        # Average over Z dimension to get 2D
        probs_2d = probs.mean(dim=2)  # (B, 1, 4, 4)

        # Move to CPU and process each prediction
        probs_2d = probs_2d.cpu().numpy()

        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            # Convert to output coordinates
            x1_out = x1 // scale_factor
            y1_out = y1 // scale_factor
            x2_out = x2 // scale_factor
            y2_out = y2 // scale_factor

            # Add prediction to output
            pred_sum[y1_out:y2_out, x1_out:x2_out] += probs_2d[i, 0]
            pred_count[y1_out:y2_out, x1_out:x2_out] += 1

    # Average predictions
    mask_pred = np.divide(pred_sum, pred_count,
                          out=np.zeros_like(pred_sum),
                          where=pred_count > 0)

    return mask_pred


def main():
    # Configuration
    checkpoint_path = "path/to/your/checkpoint.ckpt"
    fragment_id = "20231005123336"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading model...")
    model = InkDetectionModel.load_from_checkpoint(checkpoint_path, strict=False)
    model.to(device)
    model.eval()

    # Get chunks from fragment
    print(f"Processing fragment {fragment_id}...")
    chunks, xyxys, output_shape = get_fragment_chunks(
        CFG.zarr_path, fragment_id, CFG.masks_path
    )

    if chunks is None:
        print("Failed to load fragment")
        return

    print(f"Found {len(chunks)} valid chunks")

    # Create dataset and dataloader
    dataset = InferenceDataset(chunks, xyxys)
    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    # Run inference
    mask_pred = run_inference(model, dataloader, output_shape, device)

    # Save result
    output_path = f"{fragment_id}_ink_prediction.png"
    mask_pred_uint8 = (np.clip(mask_pred, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_pred_uint8)
    print(f"Saved prediction to {output_path}")

    # Optional: save as numpy for further processing
    np.save(f"{fragment_id}_ink_prediction.npy", mask_pred)


if __name__ == "__main__":
    main()
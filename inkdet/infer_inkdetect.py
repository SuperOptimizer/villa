import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import zarr
import cv2
from tqdm import tqdm

# Import your model classes (from the training script)
from train_inkdetect import InkDetectionModel, CHUNK_SIZE, STRIDE, ISO_THRESHOLD, ZARR_PATH, OUTPUT_SIZE, MASKS_PATH, \
    BATCH_SIZE, NUM_WORKERS, preprocess_chunk


class InferenceDataset(Dataset):
    """Lazy-loading dataset for inference chunks"""

    def __init__(self, zarr_store, fragment_id, xyxys):
        self.zarr_store = zarr_store
        self.fragment_id = fragment_id
        self.xyxys = xyxys
        self.frag_data = self.zarr_store[fragment_id]

    def __len__(self):
        return len(self.xyxys)

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.xyxys[idx]

        # Load chunk on-demand
        chunk_3d = self.frag_data[:, y1:y2, x1:x2].astype(np.float32)
        chunk_3d = preprocess_chunk(chunk_3d)

        chunk_tensor = torch.from_numpy(chunk_3d).float()

        return chunk_tensor, torch.tensor([x1, y1, x2, y2])


def get_valid_chunk_coords(zarr_path, fragment_id, masks_path):
    """Get coordinates of valid chunks without loading data"""
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

    # Collect valid chunk coordinates only
    xyxys = []

    for y in range(0, h - CHUNK_SIZE, STRIDE):
        for x in range(0, w - CHUNK_SIZE, STRIDE):
            # Check if chunk is valid (inside fragment mask)
            chunk_mask = frag_mask[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
            if np.all(chunk_mask > 0):
                xyxys.append([x, y, x + CHUNK_SIZE, y + CHUNK_SIZE])

    return zarr_store, xyxys, (h, w)


@torch.no_grad()
def run_inference(model, dataloader, output_shape, device):
    """Run inference and assemble 2D output"""
    # Initialize output arrays
    scale_factor = CHUNK_SIZE // OUTPUT_SIZE  # 64/4 = 16

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

        for i in range(len(xyxys)):
            x1, y1, x2, y2 = xyxys[i].tolist() if hasattr(xyxys[i], 'tolist') else xyxys[i]
            # Convert to output coordinates
            x1_out = int(x1 // scale_factor)
            y1_out = int(y1 // scale_factor)
            x2_out = int(x2 // scale_factor)
            y2_out = int(y2 // scale_factor)

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
    checkpoint_path = "/vesuvius/inkdet_outputs/best_volumetric_resnet10_epoch=40.ckpt"
    fragment_id = "20231005123336"
    device = torch.device('cuda')

    # Load model
    print("Loading model...")
    model = InkDetectionModel.load_from_checkpoint(checkpoint_path, strict=False)
    model.to(device)
    model = torch.compile(model,fullgraph=True, dynamic=False)

    model.eval()

    # Get chunk coordinates (not loading data yet)
    print(f"Processing fragment {fragment_id}...")
    zarr_store, xyxys, output_shape = get_valid_chunk_coords(
        ZARR_PATH, fragment_id, MASKS_PATH
    )

    if xyxys is None:
        print("Failed to load fragment")
        return

    print(f"Found {len(xyxys)} valid chunks")

    # Create dataset with lazy loading
    dataset = InferenceDataset(zarr_store, fragment_id, xyxys)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
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
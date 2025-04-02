#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import zarr
import json
import os
import fsspec
from matplotlib.widgets import Button
import random


class ZarrRater:
    def __init__(self, img_path, lbl_path, chunk_size=128, label_threshold=0.05, output_json="ratings.json"):
        """
        Tool for rating zarr chunks with overlaid labels.

        Args:
            img_path: Path to image zarr (can be http)
            lbl_path: Path to label zarr (can be http)
            chunk_size: Size of chunks to display (default: 128)
            label_threshold: Minimum percentage of chunk that must contain label (default: 0.05)
            output_json: Path to save ratings (default: ratings.json)
        """
        self.chunk_size = chunk_size
        self.label_threshold = label_threshold
        self.output_json = output_json
        self.ratings = []

        # --- Opening the image zarr ---
        print(f"Opening image zarr root: {img_path}")
        try:
            if img_path.startswith('http://') or img_path.startswith('https://'):
                mapper = fsspec.get_mapper(img_path)
                root = zarr.open(mapper, mode='r')
                # Access group "0" from the root
                self.img_zarr = root['0']
                print(f"Successfully opened image zarr group '0' with shape: {self.img_zarr.shape}")
            else:
                root = zarr.open(img_path, mode='r')
                self.img_zarr = root['0'] if '0' in root else root
                print(f"Successfully opened local image zarr with shape: {self.img_zarr.shape}")
        except Exception as e:
            print(f"Error opening image zarr: {e}")
            raise ValueError(f"Failed to open image zarr at {img_path}. Error: {e}")

        # --- Opening the label zarr ---
        print(f"Opening label zarr root: {lbl_path}")
        try:
            if lbl_path.startswith('http://') or lbl_path.startswith('https://'):
                fs = fsspec.filesystem('http')
                mapper = fs.get_mapper(lbl_path.rstrip('/'))
                root = zarr.open(mapper, mode='r')
                self.lbl_zarr = root['0']
                print(f"Successfully opened label zarr group '0' with shape: {self.lbl_zarr.shape}")
            else:
                root = zarr.open(lbl_path, mode='r')
                self.lbl_zarr = root['0'] if '0' in root else root
                print(f"Successfully opened local label zarr with shape: {self.lbl_zarr.shape}")
        except Exception as e:
            print(f"Error opening label zarr: {e}")
            raise ValueError(f"Failed to open label zarr at {lbl_path}. Error: {e}")


        self.depth, self.height, self.width = self.img_zarr.shape[0:3]
        print(f"Zarr dimensions: {self.depth} x {self.height} x {self.width}")

        # Load existing ratings if file exists
        if os.path.exists(self.output_json):
            with open(self.output_json, 'r') as f:
                self.ratings = json.load(f)
            print(f"Loaded {len(self.ratings)} existing ratings")

        # Setup plotting
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Next button
        self.next_button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.next_button = Button(self.next_button_ax, 'Next')
        self.next_button.on_clicked(self._next_chunk)

        # Skip button
        self.skip_button_ax = plt.axes([0.6, 0.01, 0.1, 0.05])
        self.skip_button = Button(self.skip_button_ax, 'Skip')
        self.skip_button.on_clicked(self._next_chunk)

        # Initialize
        self.current_coords = None
        self.current_chunk_img = None
        self.current_chunk_lbl = None
        self.find_next_chunk()

    def find_next_chunk(self):
        """Find next chunk with sufficient label content"""
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            # Randomly select coordinates
            z = random.randint(0, self.depth - self.chunk_size)
            y = random.randint(0, self.height - self.chunk_size)
            x = random.randint(0, self.width - self.chunk_size)

            # Check if we've already rated this location
            if any(r['coords'] == [z, y, x] for r in self.ratings):
                attempts += 1
                continue

            # Extract chunk and check label content
            lbl_chunk = self.lbl_zarr[z:z + self.chunk_size, y:y + self.chunk_size, x:x + self.chunk_size]
            label_ratio = np.count_nonzero(lbl_chunk) / lbl_chunk.size

            if label_ratio >= self.label_threshold:
                self.current_coords = [z, y, x]
                self.current_chunk_img = self.img_zarr[z:z + self.chunk_size, y:y + self.chunk_size,
                                         x:x + self.chunk_size]
                self.current_chunk_lbl = lbl_chunk
                self.display_current_chunk()
                return True

            attempts += 1

        print("Could not find a suitable chunk after multiple attempts")
        return False

    def display_current_chunk(self):
        """Display current chunk with label overlay"""
        if self.current_chunk_img is None or self.current_chunk_lbl is None:
            return

        # Take middle slice for 3D data
        mid_z = self.chunk_size // 2
        img_slice = self.current_chunk_img[mid_z]
        lbl_slice = self.current_chunk_lbl[mid_z]

        # Calculate label percentage for this slice
        label_percentage = np.count_nonzero(lbl_slice) / lbl_slice.size * 100

        # Normalize image for display
        if img_slice.max() > 0:
            img_norm = img_slice / img_slice.max()
        else:
            img_norm = img_slice

        # Clear previous display
        self.ax.clear()
        self.ax.imshow(img_norm, cmap='gray')

        # Overlay label with transparency
        lbl_mask = np.ma.masked_where(lbl_slice == 0, lbl_slice)
        self.ax.imshow(lbl_mask, cmap='jet', alpha=0.5)

        # Add coordinate info and status
        z, y, x = self.current_coords
        self.ax.set_title(f"Coords: Z={z + mid_z}, Y={y}, X={x} | Label: {label_percentage:.1f}%")

        # Add rating instruction at the bottom
        self.fig.text(0.5, 0.01, f"Press 1-10 to rate | {len(self.ratings)} ratings saved",
                      ha='center', fontsize=12)
        plt.draw()

    def _on_key(self, event):
        """Handle key press events for rating input"""
        if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            try:
                rating = float(event.key)
                self.save_rating(rating)
            except ValueError:
                pass
        elif event.key == '0':  # 0 key is used for rating 10
            self.save_rating(10)
        elif event.key == 'enter':
            self.find_next_chunk()

    def _next_chunk(self, event):
        """Handle next button click"""
        self.find_next_chunk()

    def save_rating(self, rating):
        """Save current rating to the ratings list and JSON file"""
        if rating < 0 or rating > 10:
            print("Rating must be between 0 and 10")
            return

        z, y, x = self.current_coords
        entry = {
            "coords": [z, y, x],
            "rating": rating,
            "chunk_size": self.chunk_size
        }

        self.ratings.append(entry)

        # Save to JSON file
        with open(self.output_json, 'w') as f:
            json.dump(self.ratings, f, indent=2)

        print(f"Saved rating {rating} for coords {self.current_coords}")
        self.find_next_chunk()

    def run(self):
        """Main loop for the rater"""
        print("=" * 50)
        print("Zarr Chunk Rater Instructions:")
        print("=" * 50)
        print("- Press 1-10 to rate the current chunk (1=worst, 10=best)")
        print("- Press Enter or click 'Next' to move to next chunk")
        print("- Click 'Skip' to skip the current chunk without rating")
        print("- Ratings are automatically saved to:", self.output_json)
        print("=" * 50)
        plt.show()


if __name__ == "__main__":
    # Configure these parameters
    img_path = "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes_zarr/20231117161658.zarr/"
    lbl_path = "https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s4/surfaces/s4_059_medial_ome.zarr/"

    chunk_size = 128  # Size of chunks to display
    label_threshold = 0.05  # Minimum percentage of chunk that must contain label
    output_json = "ratings.json"  # Output JSON file for ratings

    rater = ZarrRater(
        img_path,
        lbl_path,
        chunk_size=chunk_size,
        label_threshold=label_threshold,
        output_json=output_json
    )

    rater.run()

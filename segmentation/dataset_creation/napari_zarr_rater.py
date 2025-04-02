#!/usr/bin/env python

import numpy as np
import zarr
import json
import os
import fsspec
import random
import napari
from qtpy.QtWidgets import QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QWidget, QSizePolicy

class NapariZarrRater:
    def __init__(self, img_path, lbl_path, chunk_size=128, label_threshold=0.01, output_json="ratings.json"):
        self.chunk_size = chunk_size
        self.label_threshold = label_threshold
        self.output_json = output_json
        self.ratings = []

        # Open image zarr
        mapper = fsspec.get_mapper(img_path)
        self.img_zarr = zarr.open(mapper, mode='r')['0']

        # Open label zarr
        mapper_lbl = fsspec.get_mapper(lbl_path)
        self.lbl_zarr = zarr.open(mapper_lbl, mode='r')['0']

        self.depth, self.height, self.width = self.img_zarr.shape[0:3]

        if os.path.exists(self.output_json):
            with open(self.output_json, 'r') as f:
                self.ratings = json.load(f)

        self.current_coords = None
        self.viewer = napari.Viewer(ndisplay=2)
        self.image_layer = None
        self.label_layer = None

        self.setup_ui()
        self.find_next_chunk()

    def setup_ui(self):
        self.status_widget = QWidget()
        layout = QVBoxLayout()
        self.status_widget.setLayout(layout)

        self.status_label = QLabel("Ready to rate chunks")
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        good_btn = QPushButton("good")
        decent_btn = QPushButton("decent")
        bad_btn = QPushButton("bad")
        good_btn.clicked.connect(lambda: self.save_rating(2))
        decent_btn.clicked.connect(lambda: self.save_rating(1))
        bad_btn.clicked.connect(lambda: self.save_rating(0))
        button_layout.addWidget(good_btn)
        button_layout.addWidget(decent_btn)
        button_layout.addWidget(bad_btn)
        layout.addLayout(button_layout)

        nav_layout = QHBoxLayout()
        next_btn = QPushButton("Next Chunk")
        next_btn.clicked.connect(self.find_next_chunk)
        skip_btn = QPushButton("Skip")
        skip_btn.clicked.connect(self.find_next_chunk)
        nav_layout.addWidget(next_btn)
        nav_layout.addWidget(skip_btn)
        layout.addLayout(nav_layout)

        self.viewer.window.add_dock_widget(
            self.status_widget,
            name="Rating Controls",
            area="bottom"
        )

        @self.viewer.bind_key('g')
        def good(viewer):
            self.save_rating(2)

        @self.viewer.bind_key('d')
        def decent(viewer):
            self.save_rating(1)

        @self.viewer.bind_key('b')
        def bad(viewer):
            self.save_rating(0)

        @self.viewer.bind_key('Enter')
        def next_chunk(viewer):
            self.find_next_chunk()

    def find_next_chunk(self):
        attempts = 0
        while attempts < 100:
            z = random.randint(0, self.depth - self.chunk_size)
            y = random.randint(0, self.height - self.chunk_size)
            x = random.randint(0, self.width - self.chunk_size)

            if any(r['coords'] == [z, y, x] for r in self.ratings):
                attempts += 1
                continue

            lbl_chunk = self.lbl_zarr[z:z + self.chunk_size, y:y + self.chunk_size, x:x + self.chunk_size]
            if np.count_nonzero(lbl_chunk) / lbl_chunk.size >= self.label_threshold:
                self.current_coords = [z, y, x]
                img_chunk = self.img_zarr[z:z + self.chunk_size, y:y + self.chunk_size, x:x + self.chunk_size]
                self.display_chunk(img_chunk, lbl_chunk)
                return
            attempts += 1

        self.status_label.setText("No suitable chunk found.")

    def display_chunk(self, img_chunk, lbl_chunk):
        img_norm = img_chunk / img_chunk.max() if img_chunk.max() > 0 else img_chunk

        if self.image_layer is None:
            self.image_layer = self.viewer.add_image(img_norm, name='Image')
        else:
            self.image_layer.data = img_norm

        if self.label_layer is None:
            self.label_layer = self.viewer.add_labels(lbl_chunk, name='Labels', opacity=0.5)
        else:
            self.label_layer.data = lbl_chunk

        z, y, x = self.current_coords
        self.status_label.setText(f"Coords: Z={z}, Y={y}, X={x} | Ratings: {len(self.ratings)}")
        self.viewer.reset_view()

    def save_rating(self, rating):
        if self.current_coords is None:
            return

        entry = {
            "coords": self.current_coords,
            "rating": rating,
            "chunk_size": self.chunk_size
        }
        self.ratings.append(entry)

        with open(self.output_json, 'w') as f:
            json.dump(self.ratings, f, indent=2)

        self.find_next_chunk()

    def run(self):
        napari.run()

if __name__ == "__main__":
    img_path = "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes_zarr/20231117161658.zarr/"
    lbl_path = "https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s4/surfaces/s4_059_medial_ome.zarr/"

    rater = NapariZarrRater(img_path, lbl_path, chunk_size=256)
    rater.run()

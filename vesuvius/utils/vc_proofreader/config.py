# config.py

config = {
    # Zarr paths can be local file system paths or HTTP URLs.
    "image_zarr": "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr/20241024131838.zarr",  # Replace with your image zarr path
    "label_zarr": "/mnt/raid_nvme/ced_volumes/s5_paths_vx_ced_raw_ome.zarr",    # Replace with your label zarr path

    # Output directory where subfolders "imagesTr" and "labelsTr" will be created.
    "dataset_out_path": "/mnt/raid_nvme/ced_labels/s5/",

    # Patch extraction settings (patch_size is specified in display resolution units).
    "patch_size": 256,
    "sampling": "sequence",  # Options: "sequence" or "random"

    # Progress saving options.
    "save_progress": True,                                    # Set to True to enable saving progress.
    "progress_file": "/mnt/raid_nvme/esrf_labels/s5/progress.json",                # Path to the file where progress is saved.
    
    # Output zarr for approved labels
    "output_label_zarr": "/mnt/raid_nvme/esrf_labels/s5/labels.zarr",      # Path for the output label zarr (binary uint8)
}

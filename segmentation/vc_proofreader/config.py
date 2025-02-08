# config.py

config = {
    # Zarr paths can be local file system paths or HTTP URLs.
    "image_zarr": "https://dl.ash2txt.org/community-uploads/james/Scroll1/Scroll1_8um.zarr/",  # Replace with your image zarr path
    "label_zarr": "/mnt/raid_hdd/labels/surfaces/s1_surface_label.zarr",    # Replace with your label zarr path

    # Output directory where subfolders "imagesTr" and "labelsTr" will be created.
    "dataset_out_path": "/mnt/raid_nvme/testds",

    # Patch extraction settings (patch_size is specified in display resolution units).
    "patch_size": 128,
    "sampling": "sequence",  # Options: "sequence" or "random"

    # Progress saving options.
    "save_progress": True,                                    # Set to True to enable saving progress.
    "progress_file": "/mnt/raid_nvme/testds/progress.txt",                # Path to the file where progress is saved.
}

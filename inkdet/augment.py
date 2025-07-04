import numpy as np
import random
import skimage
from scipy.ndimage import map_coordinates, gaussian_filter, rotate, convolve1d
from scipy.interpolate import RegularGridInterpolator


class VolumetricAugmentations:

    def __init__(self, augment_chance):
        self.augment_chance = augment_chance
        pass

    def elastic_transform_3d(self, volume, mask, alpha=500, sigma=20):
        if random.random() < .1:
            shape = volume.shape
            random_state = np.random.RandomState(None)
            dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                                  np.arange(shape[2]), indexing='ij')
            indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            volume = map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
            mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
        return volume, mask

    def grid_distortion_3d(self, volume, mask, num_steps=5, distort_limit=0.3):
        if random.random() < .1:
            shape = volume.shape
            grid_z = np.linspace(0, shape[0] - 1, num_steps)
            grid_y = np.linspace(0, shape[1] - 1, num_steps)
            grid_x = np.linspace(0, shape[2] - 1, num_steps)
            distort_z = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))
            distort_y = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))
            distort_x = np.random.uniform(-distort_limit, distort_limit, (num_steps, num_steps, num_steps))
            z_coords = np.arange(shape[0])
            y_coords = np.arange(shape[1])
            x_coords = np.arange(shape[2])
            f_z = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_z)
            f_y = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_y)
            f_x = RegularGridInterpolator((grid_z, grid_y, grid_x), distort_x)
            zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            points = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
            dz = f_z(points).reshape(shape) * shape[0]
            dy = f_y(points).reshape(shape) * shape[1]
            dx = f_x(points).reshape(shape) * shape[2]
            indices = np.reshape(zz + dz, (-1, 1)), np.reshape(yy + dy, (-1, 1)), np.reshape(xx + dx, (-1, 1))
            volume = map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
            mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)
        return volume, mask

    def anisotropic_gaussian_blur_3d(self, volume):
        if random.random() < self.augment_chance:
            sigma_z = random.uniform(0.5, 1.0)
            sigma_xy = random.uniform(0.5, 1.0)
            volume = gaussian_filter(volume, sigma=(sigma_z, sigma_xy, sigma_xy))
        return volume

    def random_gamma_3d(self, volume, gamma_limit=(0.8, 1.2)):
        if random.random() < self.augment_chance:
            gamma = random.uniform(gamma_limit[0], gamma_limit[1])
            volume = np.clip(volume, 1e-7, 1)
            volume = np.power(volume, gamma)
        return volume

    def random_flip_3d(self, volume, mask):
        if random.random() < self.augment_chance:
            volume = np.flip(volume, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < self.augment_chance:
            volume = np.flip(volume, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < self.augment_chance:
            volume = np.flip(volume, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return volume, mask

    def random_rotation_3d(self, volume, mask):
        if random.random() < self.augment_chance:
            angle_x = random.uniform(-180, 180)
            angle_y = random.uniform(-180, 180)
            angle_z = random.uniform(-180, 180)
            volume = rotate(volume, angle_x, axes=(1, 2), reshape=False, order=1)
            volume = rotate(volume, angle_y, axes=(0, 2), reshape=False, order=1)
            volume = rotate(volume, angle_z, axes=(0, 1), reshape=False, order=1)
            mask = rotate(mask, angle_x, axes=(1, 2), reshape=False, order=0)
            mask = rotate(mask, angle_y, axes=(0, 2), reshape=False, order=0)
            mask = rotate(mask, angle_z, axes=(0, 1), reshape=False, order=0)
        return volume, mask

    def random_brightness_contrast_3d(self, volume):
        if random.random() < self.augment_chance:
            brightness = random.uniform(-0.2, 0.2)
            volume = volume + brightness
            contrast = random.uniform(0.8, 1.2)
            mean = np.mean(volume)
            volume = (volume - mean) * contrast + mean
        return volume

    def random_intensity_shift_3d(self, volume):
        if random.random() < self.augment_chance:
            z_size, y_size, x_size = volume.shape
            block_size = 16
            shift_map = np.random.uniform(-0.1, 0.1,
                                          (z_size // block_size + 1, y_size // block_size + 1,
                                           x_size // block_size + 1))
            shift_map = np.repeat(np.repeat(np.repeat(shift_map, block_size, axis=0),
                                            block_size, axis=1), block_size, axis=2)
            shift_map = shift_map[:z_size, :y_size, :x_size]
            volume = volume + shift_map
        return volume

    def motion_blur_z_axis(self, volume):
        if random.random() < self.augment_chance:
            kernel_size = random.choice([3, 5, 7])
            kernel = np.ones(kernel_size) / kernel_size
            volume = convolve1d(volume, kernel, axis=0, mode='reflect')
        return volume

    def gradient_based_dropout(self, volume):
        if random.random() < self.augment_chance:
            gz = np.abs(np.diff(volume, axis=0, prepend=volume[0:1]))
            gy = np.abs(np.diff(volume, axis=1, prepend=volume[:, 0:1]))
            gx = np.abs(np.diff(volume, axis=2, prepend=volume[:, :, 0:1]))
            gradient_mag = gz + gy + gx
            threshold = np.percentile(gradient_mag, random.uniform(70, 90))
            mask = gradient_mag > threshold
            if random.random() < 0.5:
                volume[mask] *= random.uniform(0.3, 0.7)
        return volume

    def random_noise_3d(self, volume):
        if random.random() < self.augment_chance:
            noise_type = random.choice(['gaussian', 'anisotropic_blur'])
            if noise_type == 'gaussian':
                noise = np.random.normal(0, random.uniform(0.01, 0.05), volume.shape)
                volume = volume + noise
            else:
                volume = self.anisotropic_gaussian_blur_3d(volume)
        return volume

    def coarse_dropout_3d(self, volume):
        if random.random() < self.augment_chance:
            h, w, d = volume.shape
            n_holes = random.randint(1, 3)
            for _ in range(n_holes):
                hole_size = int(0.2 * min(h, w, d))
                x = random.randint(0, h - hole_size)
                y = random.randint(0, w - hole_size)
                z = random.randint(0, d - hole_size)
                volume[x:x + hole_size, y:y + hole_size, z:z + hole_size] = 0
        return volume

    def apply(self, volume, mask):
        volume, mask = self.random_flip_3d(volume, mask)
        volume, mask = self.random_rotation_3d(volume, mask)
        volume, mask = self.elastic_transform_3d(volume, mask)
        volume, mask = self.grid_distortion_3d(volume, mask)

        volume = self.random_brightness_contrast_3d(volume)
        volume = self.random_gamma_3d(volume)
        volume = self.random_intensity_shift_3d(volume)
        volume = self.motion_blur_z_axis(volume)
        volume = self.gradient_based_dropout(volume)
        volume = self.random_noise_3d(volume)
        volume = self.coarse_dropout_3d(volume)

        volume = skimage.exposure.equalize_hist(volume)
        volume = np.clip(volume, 0, 1)
        mask = np.clip(mask, 0, 1)
        return volume, mask
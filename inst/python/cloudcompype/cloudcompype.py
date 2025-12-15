"""
CloudComPype: A pipeline-ready point cloud processing tool based on CloudComPy.

This module wraps the 3D point cloud processing functions of the CloudComPy
library in a simple class. It enables the creation of easy-to-read pipelines
for scientific point cloud processing, where documentation is necessary.

Functions:
    voxelize_coords: Snap point coordinates to a regular grid
    gini: Calculate Gini coefficient for inequality measurement

Classes:
    dtm_config: Configuration dataclass for DTM extraction
    cloudpy: Main wrapper class for CloudComPy point cloud operations
"""

import os
import statistics as stats
import math
from warnings import warn
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utils import *


class cloudcompype:
    """
    A wrapper class for CloudComPy point cloud objects with convenience methods
    for common point cloud processing tasks, to ease the creation of analysis
    pipelines.

    Attributes:
        pointcloud: The underlying CloudComPy ccPointCloud object
    """

    def __init__(self, pointcloud):
        """
        Initialize the cloudcompype object with a CloudComPy point cloud.

        Args:
            pointcloud: A CloudComPy ccPointCloud object

        Raises:
            TypeError: If pointcloud is not a valid CloudComPy PointCloud
                       object
        """
        if not isinstance(pointcloud, cc.ccPointCloud):
            raise TypeError("pointcloud must be a ccPointCloud object!")
        self.pointcloud = pointcloud

    def save_cloud(self, path, verbose=True):
        """
        Save the current pointcloud within cloudcompype to a file.

        Args:
            path (str): File path where to save the point cloud
            verbose (bool): If True, print file information after saving
        """
        cc.SavePointCloud(self.pointcloud, path)
        if verbose == True:
            abs_path = os.path.abspath(path)
            size = os.path.getsize(abs_path)
            num_pts = self.pointcloud.size()
            print(f'Saved Pointcloud at: {abs_path}.')
            print(f'Points: {num_pts}')
            print(f'Size: {size} B')

    def to_np_array(self):
        """
        Convert the point cloud to a NumPy array.

        Returns:
            np.ndarray: A copy of the point cloud coordinates as a NumPy array
        """
        return self.pointcloud.toNpArrayCopy()

    def update_from_np_array(self, array):
        """
        Update the pointcloud within cloudcompype from a NumPy array.

        Args:
            array (np.ndarray): The array of points (N x 3) to update the 
                                pointcloud with
        """
        self.pointcloud.coordsFromNPArray_copy(array)

    def delete_sf(self, index=None, all=False):
        """
        Delete scalar fields from the pointcloud.

        Args:
            index (int, optional): Index of the scalar field to delete.
                                   Ignored if all=True.
            all (bool): If True, delete all scalar fields.
        """
        if all:
            if index is not None:
                warn(
                    "all == True. All scalar fields will be deleted! index"
                    "will be ignored!"
                )
            self.pointcloud.deleteAllScalarFields()
        else:
            self.pointcloud.deleteScalarField(index)

    def delete_rgb(self):
        """
        Delete RGB color information from the pointcloud.
        """
        self.pointcloud.unallocateColors()

    def sor(self, n_pts, n_sigma):
        """
        Apply Statistical Outlier Removal (SOR) filtering to the point cloud.

        Args:
            n_pts (int): Number of neighboring points to consider
            n_sigma (float): Standard deviation threshold

        Returns:
            cloudpy: The filtered cloudpy object (self)
        """
        ref_cloud = cc.CloudSamplingTools.sorFilter(self.pointcloud,
                                                    knn=n_pts,
                                                    nSigma=n_sigma)
        self.pointcloud, res = self.pointcloud.partialClone(ref_cloud)
        return self

    def noise(self, radius=0.1, nSigma=1, abs_error_use=True, abs_error_val=1, remove_iso_points=True):
        """
        Apply noise filtering to remove isolated points.

        Args:
            radius (float): Kernel radius for filtering
            nSigma (float): Standard deviation threshold
            abs_error_use (bool): Whether to use absolute error threshold
            abs_error_val (float): Absolute error threshold value
            remove_iso_points (bool): Whether to remove isolated points

        Returns:
            cloudpy: The filtered cloudpy object (self)
        """
        ref_cloud = cc.CloudSamplingTools.noiseFilter(self.pointcloud,
                                                      kernelRadius=radius,
                                                      nSigma=nSigma,
                                                      removeIsolatedPoints=remove_iso_points,
                                                      useAbsoluteError=abs_error_use,
                                                      absoluteError=abs_error_val)
        self.pointcloud, res = self.pointcloud.partialClone(ref_cloud)
        return self

    def crop_xy(self, dim, center="origin"):
        """
        Crop the point cloud in the XY plane to a square region.

        Args:
            dim (float): Dimension (width/height) of the crop box
            center (str or list): Center of the crop. Can be 'origin', 'center',
                or [x, y] coordinates. 'origin' uses [0,0], 'center' uses the
                geometric center of the point cloud's bounding box.

        Returns:
            cloudpy: The cropped cloudpy object (self)
        """
        bb = self.pointcloud.getOwnBB()
        if type(center) != list:
            if center == "origin":
                center = [0, 0, 0]
            if center == "center":
                center = [(bb.minCorner()[0] + bb.maxCorner()[0]) / 2,
                          (bb.minCorner()[1] + bb.maxCorner()[1]) / 2]
        minCorner = [center[0] - dim / 2,
                     center[1] - dim / 2,
                     bb.minCorner()[2]]
        maxCorner = [center[0] + dim / 2,
                     center[1] + dim / 2,
                     bb.maxCorner()[2]]
        bb_crop = cc.ccBBox(minCorner, maxCorner, True)
        self.pointcloud = cc.ExtractSlicesAndContours(entities=[self.pointcloud],
                                                      bbox=bb_crop)[0][0]
        return self

    def crop_z(self, dim, direction):
        """
        Crop the point cloud along the Z-axis (height).

        Args:
            dim (float): Thickness of the Z crop
            direction (str): Direction to crop, either 'up' (keep upper portion)
                or 'down' (keep lower portion)
        """
        bb = self.pointcloud.getOwnBB()
        bottom = bb.minCorner()[2]
        top = bb.maxCorner()[2]
        if direction == "up":
            new_bottom = bottom + dim
            new_top = top
        if direction == "down":
            new_bottom = bottom
            new_top = top - dim
        minCorner = [bb.minCorner()[0],
                     bb.minCorner()[1],
                     new_bottom]
        maxCorner = [bb.maxCorner()[0],
                     bb.maxCorner()[1],
                     new_top]
        bb_crop = cc.ccBBox(minCorner, maxCorner, True)
        self.pointcloud = cc.ExtractSlicesAndContours(entities=[self.pointcloud],
                                                      bbox=bb_crop)[0][0]

    def voxelize(self, res, modify_self=True):
        """
        Voxelize the point cloud by snapping points to a 3D grid.

        Args:
            res (float): Voxel resolution.
            modify_self (bool): If True, modifies the current point cloud. If False, returns a new cloudpy object.

        Returns:
            cloudpy: The voxelized cloudpy object (either self or a new instance).
        """
        if modify_self:
            target = self.pointcloud
        else:
            target = self.pointcloud.cloneThis()
        cloud_coords = target.toNpArrayCopy()
        cloud_coords_vox = voxelize_coords(cloud_coords, res)
        target.coordsFromNPArray_copy(cloud_coords_vox)

        if modify_self:
            return self
        else:
            return cloudpy(target)

    def extract_dtm(self, config: dtm_config, output="mesh"):
        """
        Extract Digital Terrain Model (DTM) from the point cloud.

        Args:
            config (dtm_config): Configuration object with DTM extraction parameters
            output (str): Output type, either "mesh" or "pointcloud"

        Returns:
            cc.ccMesh or cc.ccPointCloud: The extracted DTM as mesh or point cloud

        Raises:
            ValueError: If output is not "mesh" or "pointcloud"
        """
        if output not in ["mesh", "pointcloud"]:
            raise ValueError(r'output must be either "mesh" or "pointcloud"!')
        dtm_raw = cc.RasterizeToCloud(
            cloud=self.pointcloud, gridStep=config.resolution, projectionType=cc.ProjectionType.PROJ_MINIMUM_VALUE)
        dtm_filtered_ref = cc.CloudSamplingTools.sorFilter(
            cloud=dtm_raw, knn=config.n_pts_sor, nSigma=config.n_sigma_sor)
        dtm_filtered = dtm_raw.partialClone(dtm_filtered_ref)[0]
        dtm_mesh = cc.ccMesh.triangulate(
            cloud=dtm_filtered, type=cc.TRIANGULATION_TYPES.DELAUNAY_2D_BEST_LS_PLANE)
        dtm_mesh.laplacianSmooth(
            nbIteration=config.i_smooth, factor=config.f_smooth)
        if output == "mesh":
            dtm = dtm_mesh
        else:
            dtm_sampled = dtm_mesh.samplePoints(
                densityBased=True, samplingParameter=config.n_pts_sampling, withRGB=False, withTexture=False)
            dtm = cc.RasterizeToCloud(
                cloud=dtm_sampled, gridStep=config.dtm_res, emptyCellFillStrategy=cc.EmptyCellFillOption.INTERPOLATE_DELAUNAY, customHeight=-10000)
            dtm = dtm.filterPointsByScalarValue(minVal=-9999, maxVal=10000)
            dtm.deleteAllScalarFields()
        return dtm

    def normalize(self, dtm, method="unvoxelized"):
        if method not in ["unvoxelized", "voxelized"]:
            raise ValueError(
                r'method must be either "unvoxelized" or "voxelized"!')

        if method == "unvoxelized":
            if not isinstance(dtm, cc.ccMesh):
                raise TypeError(
                    r'In unvoxelized normalization the dtm must be type cc.ccMesh object!')
            params = cc.Cloud2MeshDistancesComputationParams()
            cc.DistanceComputationTools.computeCloud2MeshDistances(self.pointcloud,
                                                                   dtm,
                                                                   params)
            dic = self.pointcloud.getScalarFieldDic()
            sf = self.pointcloud.getScalarField(dic['C2M absolute distances'])
            sf_np = sf.toNpArrayCopy()
            self.delete_sf(index=dic['C2M absolute distances'])
            cloud_np = self.pointcloud.toNpArrayCopy()
            cloud_np[:, 2] = sf_np
            self.update_from_np_array(cloud_np)
        else:
            if not isinstance(dtm, cc.ccPointCloud):
                raise TypeError(
                    r'In voxelized normalization the dtm must be type cc.ccPointCloud object!')
            cloud_coords = pd.DataFrame(
                self.pointcloud.toNpArrayCopy(), columns=['x', 'y', 'z'])
            dtm_coords = pd.DataFrame(
                dtm.toNpArrayCopy(), columns=['x', 'y', 'z'])
            merged_coords = pd.merge(cloud_coords, dtm_coords, on=[
                                     'x', 'y'], suffixes=('', '_dtm'))

            if len(merged_coords) == 0:
                raise ValueError("No matching coordinates found between point cloud and DTM. "
                                 "Ensure both use the same coordinate system and resolution.")

            merged_coords['z'] = merged_coords['z'] - \
                np.sign(merged_coords['z_dtm']) * merged_coords['z_dtm'].abs()
            normalized_coords = merged_coords[merged_coords['z'] >= 0].drop(columns=[
                                                                            'z_dtm'])

            if len(normalized_coords) == 0:
                warn("All points were filtered out during normalization. "
                     "This may indicate DTM heights are above all point cloud heights.")

            self.pointcloud.coordsFromNPArray_copy(
                normalized_coords.to_numpy())
        return self

    def align_to_north(self, p2, p1=[0, 0], heading='north'):
        """
        Align the point cloud to face north based on reference points.

        Args:
            p2 (list): Second reference point [x, y]
            p1 (list): First reference point [x, y] (default: [0, 0])
            heading (str): Cardinal direction to align to ('north', 'east', 'south', 'west')

        Returns:
            cloudpy: The aligned cloudpy object (self)

        Raises:
            ValueError: If heading is not a valid cardinal direction
        """
        heading = heading.lower()
        if heading not in ["north", "east", "south", "west"]:
            raise ValueError("heading is not a cardinal direction!")

        heading_vector = [p2[0] - p1[0], p2[1] - p1[1]]

        # Check for zero-length heading vector
        heading_norm = np.linalg.norm(heading_vector)
        if heading_norm == 0:
            raise ValueError(
                "Heading vector cannot be zero-length (p1 and p2 are identical)")

        y_vector = [0 - p1[0], 1 - p1[1]]
        heading_vector_u = heading_vector / heading_norm
        y_vector_u = y_vector / np.linalg.norm(y_vector)
        align_angle = np.arccos(
            np.clip(np.dot(heading_vector_u, y_vector_u), -1.0, 1.0))
        if heading == "east":
            align_angle = align_angle + 2 * np.pi
        elif heading == "south":
            align_angle = align_angle + ((3 * np.pi) / 2)
        elif heading == "west":
            align_angle = align_angle + np.pi

        align_tr = cc.ccGLMatrix()
        align_tr.initFromParameters(align_angle, (0, 0, 1), (0, 0, 0))
        self.pointcloud.applyRigidTransformation(align_tr)
        return self

    def boxdim(self, box_size, plot=False):
        """
        Estimate the box-counting dimension (fractal dimension) of the point cloud.

        Args:
            box_size (float): Maximum box size for dimension estimation
            plot (bool): Whether to create and return a plot

        Returns:
            float or tuple: Box dimension value, or (dimension, plot) if plot=True
        """
        bb = self.pointcloud.getOwnBB()
        minCorner = bb.minCorner()
        maxCorner = bb.maxCorner()
        r = np.subtract(maxCorner, minCorner).max()

        n_list = [1]
        r_list = [r]
        coords = self.pointcloud.toNpArrayCopy()

        while r > box_size:
            r = r / 2
            vox = r * np.around(a=coords / r)
            n = len(np.unique(vox, axis=0))
            n_list.append(n)
            r_list.append(r)

        log_rinv = np.log(np.divide(1, r_list))
        log_n = np.log(n_list)

        regression = stats.linear_regression(log_rinv, log_n)
        slope = regression.slope
        intercept = regression.intercept
        boxdim = round(slope, 2)

        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(log_rinv, log_n, 'o', label='Data points')
            plt.plot(log_rinv, intercept + slope * log_rinv,
                     'r-', label=f'Fit: slope = {slope:.2f}')
            plt.xlabel('log(1/r)')
            plt.ylabel('log(N)')
            plt.title('Box Dimension Estimation')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            return boxdim, plt
        else:
            return boxdim

    def enl(self, voxel_res, layer_thickness=1, plot=False):
        """
        Calculate Effective Number of Layers (ENL) metrics for vertical structure analysis.

        Args:
            voxel_res (float): Voxel resolution for voxelization
            layer_thickness (float): Thickness of each vertical layer
            plot (bool): Whether to create and return a vertical distribution plot

        Returns:
            tuple: (ENL₀, ENL₁, ENL₂) or (ENL₀, ENL₁, ENL₂, plot) if plot=True

        Note:
            ENL metrics quantify vertical complexity in forest canopies
        """
        voxel_cloud = self.voxelize(voxel_res, modify_self=False)
        total_vox = voxel_cloud.pointcloud.size()
        coords_vox = voxel_cloud.to_np_array()

        min_corner = np.round(coords_vox.min(axis=0), 1)
        max_corner = np.round(coords_vox.max(axis=0), 1)

        def generate_sequence(start, stop, step):
            seq = np.arange(start, stop, step)
            last_value = seq[-1] + step
            if np.isclose(last_value, stop) or last_value > stop:
                seq = np.append(seq, last_value)
            return seq

        enl_seq = generate_sequence(
            min_corner[2], max_corner[2], layer_thickness)
        enl0 = len(enl_seq) - 1

        weigthed_1 = []
        weigthed_2 = []
        layer_fractions = []

        for idx, i in enumerate(enl_seq):
            if idx == 0:
                z_filter = (coords_vox[:, 2] >= i) & (
                    coords_vox[:, 2] <= i + 1)
            elif idx == list(enumerate(enl_seq))[-1][0]:
                break
            else:
                z_filter = (coords_vox[:, 2] > i) & (coords_vox[:, 2] <= i + 1)
            layer = coords_vox[z_filter]
            # np.savetxt(str(i)+"_"+str(idx)+".csv", layer, delimiter=",")

            filled_vox_ratio = len(layer) / total_vox
            # This should eventually be fixed
            # 0 layers are not plausible and occur duo to floating point(groups) above the canopy

            if filled_vox_ratio == 0:
                warn("Empty layer detected! Not considering layers past here. "
                     "This may indicate floating-point precision issues or data gaps.")
                enl_seq = enl_seq[0:idx]
                break
            else:
                weigthed_1.append(filled_vox_ratio *
                                  math.log(filled_vox_ratio))
                weigthed_2.append(filled_vox_ratio**2)
                layer_fractions.append(filled_vox_ratio)

        enl1 = math.exp(-sum(weigthed_1))
        enl2 = 1 / sum(weigthed_2)

        if plot:
            y_labels = [enl_seq[i+1] for i in range(len(enl_seq) - 1)]
            plt.figure(figsize=(5, 6))
            plt.barh(y_labels, layer_fractions, height=layer_thickness *
                     0.9, color='forestgreen', edgecolor='black')
            plt.xlabel('Fraction of filled voxels')
            plt.ylabel('Height (z)')
            plt.title('Vertical Voxel Distribution')
            plt.yticks(y_labels)
            textstr = f"ENL₀: {enl0}\nENL₁: {enl1:.2f}\nENL₂: {enl2:.2f}"
            plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
            plt.tight_layout()
            return enl0, enl1, enl2, plt
        else:
            return enl0, enl1, enl2

    def canopy_height_stats(self, res, lower_cutoff, plot=False):
        """
        Calculate canopy height statistics from rasterized point cloud.

        Args:
            res (float): Rasterization resolution
            lower_cutoff (float): Minimum height threshold for canopy analysis
            plot (bool): Whether to create and return a canopy height raster plot

        Returns:
            tuple: (max_height, mean_height, std_height, cv, gini) or 
                   (max_height, mean_height, std_height, cv, gini, plot) if plot=True
        """
        rasterized_cloud = cc.RasterizeToCloud(cloud=self.pointcloud,
                                               gridStep=res,
                                               projectionType=cc.ProjectionType.PROJ_MAXIMUM_VALUE)
        rasterized_cloud_coords = rasterized_cloud.toNpArrayCopy()
        canopy_heights = rasterized_cloud_coords[:, 2]
        z_filter = canopy_heights > lower_cutoff
        canopy_heights = canopy_heights[z_filter]
        max_height = np.max(canopy_heights)
        mean_height = np.mean(canopy_heights)
        std_height = np.std(canopy_heights)
        cv = std_height / mean_height
        gi = gini(canopy_heights)

        if plot:
            coords = rasterized_cloud.toNpArrayCopy()
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]

            # Create a pivot table for imshow (assumes regular grid)
            xi = np.unique(x)
            yi = np.unique(y)
            xi.sort()
            yi.sort()

            # Map each x,y to a 2D array
            x_idx = {val: i for i, val in enumerate(xi)}
            y_idx = {val: i for i, val in enumerate(yi)}
            # Y first because imshow uses row-major
            grid = np.full((len(yi), len(xi)), np.nan)

            for i in range(len(coords)):
                row = y_idx[coords[i, 1]]
                col = x_idx[coords[i, 0]]
                grid[row, col] = coords[i, 2]

            plt.figure(figsize=(6, 5))
            cmap = plt.get_cmap("viridis")
            norm = colors.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
            plt.imshow(grid, origin="lower", cmap=cmap, norm=norm,
                       extent=[xi.min(), xi.max(), yi.min(), yi.max()])
            plt.colorbar(label='Canopy Height (m)')
            plt.title("Canopy Height Raster")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.tight_layout()
            return max_height, mean_height, std_height, cv, gi, plt
        else:
            return max_height, mean_height, std_height, cv, gi

    def clean(self):
        """
        Remove small disconnected components from the point cloud.

        Uses connected component analysis to identify and keep only the largest
        connected component, removing noise and small isolated clusters.

        Returns:
            cloudpy: The cleaned cloudpy object (self)
        """
        octree = self.pointcloud.computeOctree()
        best_Level = octree.findBestLevelForAGivenNeighbourhoodSizeExtraction(
            0.2)
        number_of_components = cc.LabelConnectedComponents(
            [self.pointcloud], best_Level)
        if number_of_components > 1:
            scalar_dict = self.pointcloud.getScalarFieldDic()
            # Check whether the label is really always 'CC labels'
            scalar = self.pointcloud.getScalarField(scalar_dict['CC labels'])
            scalar_list = scalar.toNpArrayCopy().tolist()

            # Find the most common value (main component)
            try:
                # Python 3.8+ returns ModeResult
                mode_result = stats.mode(scalar_list)
                if hasattr(mode_result, 'mode'):
                    main_group = mode_result.mode
                else:
                    main_group = mode_result[0]  # Older versions return tuple
            except AttributeError:
                # Fallback for older Python versions
                from collections import Counter
                main_group = Counter(scalar_list).most_common(1)[0][0]

            self.pointcloud = self.pointcloud.filterPointsByScalarValue(
                main_group, main_group)  # Biggest group always seems to be labeled "1"
        self.delete_sf(index=scalar_dict['CC labels'])
        return self

    def extract_slope(self, aspect=False):
        """
        Extract slope and optionally aspect from the point cloud using plane fitting.

        Args:
            aspect (bool): Whether to also calculate aspect (orientation)

        Returns:
            float or tuple: Slope in degrees, or (slope, aspect) if aspect=True

        Note:
            Fits a plane to the point cloud and calculates slope from the plane normal.
            Aspect is measured clockwise from north (0-360 degrees).
        """
        plane = cc.ccPlane.Fit(self.pointcloud)
        equation = plane.getEquation()
        a, b, c, d = equation
        slope_rad = math.atan(math.sqrt(a**2 + b**2) / abs(c))
        slope_deg = math.degrees(slope_rad)

        if aspect:
            dx = a
            dy = b

            aspect_rad = math.atan2(dx, dy)
            aspect_deg = math.degrees(aspect_rad)

            if aspect_deg < 0:
                aspect_deg += 360

            return slope_deg, aspect_deg
        else:
            return slope_deg

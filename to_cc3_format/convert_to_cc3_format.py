"""
Convert a step2point HDF5 file to the format expected by create_cc3_showers.py.

Usage:
    python convert_to_cc3_format.py input.h5 output.h5
"""

import math
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
from tqdm import tqdm

from martina_test.metadata import Metadata

# instructions on bash
# python convert_to_cc3_format.py /path/to/step2point_output.h5


def compute_boundaries_streaming(input_path, unique_ids, chunk=10_000_000):
    """Compute searchsorted boundaries without loading all of s_evt."""
    boundaries = np.zeros(len(unique_ids) + 1, dtype=np.int64)
    boundaries[-1] = -1  # sentinel, fill at end

    with h5py.File(input_path, "r") as h5:
        ds = h5["steps"]["event_id"]
        total = ds.shape[0]
        uid_ptr = 0  # pointer into unique_ids

        for lo in range(0, total, chunk):
            hi = min(lo + chunk, total)
            chunk_data = ds[lo:hi]

            # find where each remaining unique_id first appears in this chunk
            while uid_ptr < len(unique_ids):
                idx = np.searchsorted(chunk_data, unique_ids[uid_ptr])
                if idx < len(chunk_data) and chunk_data[idx] == unique_ids[uid_ptr]:
                    boundaries[uid_ptr] = lo + idx
                    uid_ptr += 1
                else:
                    break  # not in this chunk yet

        boundaries[-1] = total
    return boundaries


def get_max_hits_batch(start, end, boundaries):
    """
    boundaries[i] = first step row for event i (length n_events+1).
    No HDF5 access needed — hit count is purely derived from boundaries.
    """
    return int(np.max(boundaries[start + 1 : end + 1] - boundaries[start:end]))


def build_events_batch(input_path, start, end, boundaries, max_hits):
    """
    Builds the (end-start, max_hits, 4) chunk for events [start, end).
    Opens HDF5 fresh — no pickling, no passing giant arrays.
    boundaries[i] = first step row for event i (length n_events+1).
    """
    chunk = np.zeros((end - start, max_hits, 4), dtype=np.float32)
    with h5py.File(input_path, "r") as h5:
        s_pos_ds = h5["steps"]["position"]  # lazy dataset handle
        s_ene_ds = h5["steps"]["energy"]
        for i in range(start, end):
            lo = int(boundaries[i])
            hi = int(boundaries[i + 1])
            if lo == hi:
                continue  # empty event
            pos = s_pos_ds[lo:hi]  # shape (n_hits, 3)
            ene = s_ene_ds[lo:hi]  # shape (n_hits,)
            n = hi - lo
            chunk[i - start, :n, :3] = pos
            chunk[i - start, :n, 3] = ene
    return chunk


def build_events_wrapper(args):
    return build_events_batch(*args)


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def incident_energy(strt, end, iE):
    # get in incident energy
    e0 = []
    for i in range(strt, end):
        # if len(mcPDG[i]) > 7: continue
        tmp = np.reshape(iE[i].take([0]), (1, 1))
        e0.append(tmp)

    e0 = np.reshape(np.asarray(e0), (-1, 1))
    return e0


def split_to_layers(points, layer_bottom_pos, cell_thickness_global, percent_buffer=0.5, layer_axis=2):
    """
    Yield points by their layer

    Parameters
    ----------
    points : np.array (N, 4)
        2D array containing hits of the points,
        in coordinates (x, y, z, energy)
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    cell_thickness_global : float
        Thickness of the cells in the detector, in the radial direction
    percent_buffer : float
        Percentage beyond the thickness of the cell to include hits
        in the layer. Won't extent the layer beyond the bottom of
        the next layer.
        (default is 0.5)
    layer_axis : int
        The index of the last axis that corrisponds to
        the direction crossing through the layers.
        (default is 2)

    Yields
    ------
    np.array
        2D array containing hits of the points on this layer,
        in coordinates (x, y, z, energy)
    """

    def floors_ceilings(layer_bottom_pos, cell_thickness_global, percent_buffer=0.5):
        """
        Find top and bottom coordinates for the layers in the detector.

        Parameters
        ----------
        layer_bottom_pos : np.array
            Array of the bottom positions of the layers.
        cell_thickness_global : float
            Thickness of the cells in the detector, in the radial direction
        percent_buffer : float
            Percentage beyond the thickness of the cell to include hits
            in the layer. Won't extent the layer beyond the bottom of
            the next layer.
            (default is 0.5)

        Returns
        -------
        layer_floors : np.array
            Array of the bottom positions of the layers.
        layer_ceilings : np.array
            Array of the top positions of the layers.
        """
        # naive calculation of the layer floors and ceilings
        layer_floors = layer_bottom_pos.copy() - percent_buffer * cell_thickness_global
        layer_ceilings = layer_bottom_pos.copy() + (1 + percent_buffer) * cell_thickness_global
        # Unless the cells are thicker than the layers, (which they shouldn't be)
        # the true ceiling for each layer is the bottom of the layer plus the thickness
        true_ceilings = np.minimum((layer_bottom_pos + cell_thickness_global)[:-1], layer_bottom_pos[1:])
        # we dont' want any extention to cross the midpoint between the true
        # ceiling and the bottom of the next layer
        mid_points = 0.5 * (true_ceilings + layer_bottom_pos[1:])
        # now enforce not crossing those midpoints
        layer_floors[1:] = np.maximum(layer_floors[1:], mid_points)
        layer_ceilings[:-1] = np.minimum(layer_ceilings[:-1], mid_points)
        return layer_floors, layer_ceilings

    z_coord = points[:, layer_axis]

    layer_floors, layer_ceilings = floors_ceilings(layer_bottom_pos, cell_thickness_global, percent_buffer)

    layer_ids = np.searchsorted(layer_floors, z_coord, side="right") - 1
    return layer_ids, layer_floors, layer_ceilings
    # for floor, ceiling in zip(layer_floors, layer_ceilings):
    #     mask = (z_coord >= floor) & (z_coord < ceiling)
    #     yield points[mask]


class Transform_pointcloud:
    def __init__(self, metadata):
        self.metadata = metadata

    def get_moment_angles(self, xyz_moments):
        def radians_to_degrees(radians):
            return radians * 180 / math.pi

        p_norm_global = xyz_moments / np.linalg.norm(xyz_moments, axis=1)[..., np.newaxis]
        self.phi_global = radians_to_degrees(np.arctan2(p_norm_global[:, 1], p_norm_global[:, 0]))
        self.theta_global = radians_to_degrees(np.arccos(p_norm_global[:, 2]))
        p_norm_local = self.global_to_local_points(p_norm_global.T).T
        theta_local = radians_to_degrees(np.arccos(p_norm_local[:, 2]))
        phi_local = radians_to_degrees(np.arctan2(p_norm_local[:, 1], p_norm_local[:, 0]))

        moment_angles = {
            "theta_global": self.theta_global,
            "phi_global": self.phi_global,
            "p_norm_global": p_norm_global,
            "p_norm_local": p_norm_local,
            "theta_local": theta_local,
            "phi_local": phi_local,
        }

        return moment_angles

    def global_to_local_points(self, points):
        def rotate_x(points, angle):
            """
            Rotate 3D points around the x-axis. For right-handed coordinate systems.
            """
            angle = angle / 180.0 * np.pi
            rotation_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ]
            )
            rotated_points = rotation_matrix @ points
            return rotated_points

        def rotate_y(points, angle):
            """
            Rotate 3D points around the y-axis. For right-handed coordinate systems.
            """
            angle = angle / 180.0 * np.pi
            rotation_matrix = np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )
            rotated_points = rotation_matrix @ points
            return rotated_points

        def rotate_z(points, angle):
            """
            Rotate 3D points around the y-axis. For right-handed coordinate systems.
            """
            angle = angle / 180.0 * np.pi
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            rotated_points = rotation_matrix @ points
            return rotated_points

        return rotate_z(rotate_x(points, 90), 90)

    def rotate_and_shift(directions, points):
        new_points = np.copy(points)
        new_points[:, :, [0, 1, 2]] = points[:, :, [2, 0, 1]]

        new_directions = directions[:, [2, 0, 1]]
        new_directions /= np.sqrt(np.sum(new_directions**2, axis=1))[:, None]

        shift_amount = new_points[:, :, [2, 2]] * new_directions[:, None, :2] / new_directions[:, None, [2, 2]]
        new_points[:, :, :2] -= shift_amount

        return new_directions, new_points

    def box_selection(self, event, restrict_x=True, restrict_y=True, restrict_z=True, box_cut=[None, None, None, None]):
        """
        Find a mask for the hits that are within the detector box.

        Parameters
        ----------
        event: np.ndarray (..., 4)
            The event to restrict
        metadata: pointcloud.metadata.Metadata
            Metadata object that contains the detector geometry information
        restrict_x: bool
            Whether to restrict the event in the x direction
        restrict_y: bool
            Whether to restrict the event in the y direction
        restrict_z: bool
            Whether to restrict the event in the z direction
        box_cut: list of 4 floats: Xmin, Xmax, Ymin, Zmin
        Returns
        -------
        in_box: np.ndarray
            Mask for the hits that are within the detector box
        """
        event = np.array(event)

        Xmin = box_cut[0] if box_cut[0] is not None else self.metadata.Xmin_global
        Xmax = box_cut[1] if box_cut[1] is not None else self.metadata.Xmax_global
        Zmin = box_cut[3] if box_cut[3] is not None else self.metadata.Zmin_global
        Zmax = box_cut[2] if box_cut[2] is not None else self.metadata.Zmax_global
        Ymin = self.metadata.Ymin_global
        assert len(event.shape) == 2, "Use per event, not on whole dataset"
        in_box = np.ones(event.shape[0], dtype=bool)
        if restrict_x:
            in_box &= (event[..., 0] < Xmax) & (event[..., 0] > Xmin)
        if restrict_y:
            in_box &= event[..., 1] > Ymin
        if restrict_z:
            in_box &= (event[..., 2] < Zmax) & (event[..., 2] > Zmin)
        if np.sum(in_box) == 0:
            msg = "No hits in box; "
            if len(event) == 0:
                msg += "No hits in event. "
        return in_box

    def get_alignment_shifts(self, phi_global, theta_global):
        """
        Calculate the shifts requires to center each layer of hits on the x=0, z=0.

        Useful if you want to learn the moments of the shower seperatly, and so can afford
        to remove that data and shift the shower so that it is all along the y axis.
        This way you only learn a lateral distribution, and not also the lateral shift.

        Parameters
        ----------
        metadata: pointcloud.metadata.Metadata
            Metadata object that contains the detector geometry information
        phi_global: np.ndarray (n_events,)
            The azimuthal moment of the shower
        theta_global: np.ndarray (n_events,)
            The polar moment of the shower

        Returns
        -------
        x_shift: np.ndarray (n_events, n_layers)
            The shift required to center the hits on the x=0 axis
        z_shift: np.ndarray (n_events, n_layers)
            The shift required to center the hits on the z=0 axis
        """

        dist_to_layers = (
            self.metadata.layer_bottom_pos_global - self.metadata.gun_xyz_pos_global[1] + self.metadata.cell_thickness_global / 2
        )
        phi_global = degrees_to_radians(phi_global.reshape((-1, 1)))
        theta_global = degrees_to_radians(theta_global.reshape((-1, 1)))
        r = (dist_to_layers / np.sin(phi_global)) / np.sin(theta_global)
        x_shift = r * np.cos(phi_global) * np.sin(theta_global) - self.metadata.gun_xyz_pos_global[0]
        z_shift = r * np.cos(theta_global) + self.metadata.gun_xyz_pos_global[2]
        return x_shift, z_shift

    def digitize_and_fuzz(self, points):
        def get_layer_bins():
            # find bins for layers
            lbp = np.array(self.metadata.layer_bottom_pos_global)
            hcal_layer_gap = lbp[-1] - lbp[-2]
            bins = np.full(len(lbp) + 1, lbp[-1] + hcal_layer_gap * 2)
            min_gap = np.min(lbp[1:] - lbp[:-1])
            bins[:-1] = lbp - 0.1 * min_gap
            return bins

        def fuzz(points):
            shifts = np.random.uniform(0, 1, size=points.shape[:2])
            points[:, :, 2] += shifts
            return points

        new_points = np.zeros_like(points)
        bins = get_layer_bins()
        layers = np.digitize(points[:, :, 2], bins) - 1
        in_bounds = ~np.logical_or(layers == -1, layers == len(bins))
        to_fill = np.sort(in_bounds, axis=1)[:, ::-1]
        new_points[to_fill] = points[in_bounds]
        new_points[to_fill, 2] = layers[in_bounds]
        new_points = fuzz(new_points)
        return new_points

    def apply_transformations(
        self,
        start,
        end,
        events,
        layer_axis=1,
    ):
        T = np.zeros_like(events[:, :, 3])  # no time info in your dataset
        if self.metadata.aligne:
            x_shift, z_shift = self.get_alignment_shifts(self.phi_global, self.theta_global)

        max_hits = 0
        event_list = []
        for event_n in range(start, end):
            # if event_n % 100 == 0:
            #     print(f"{(event_n - start) / n_events:.0%}", end="\r")

            event = events[event_n]
            if events.shape[0] == 0:
                print(f"Event {event_n} has no hits, skipping.")
                return events
            t = T[event_n]
            if self.metadata.aligne:
                layer_ids, _, _ = split_to_layers(
                    event, self.metadata.layer_bottom_pos_global, self.metadata.cell_thickness_global, layer_axis=layer_axis
                )
                valid = (layer_ids >= 0) & (layer_ids < len(x_shift[event_n]))
                valid_layer_ids = layer_ids[valid]
                event[valid, 0] -= x_shift[event_n][valid_layer_ids]
                event[valid, 2] -= z_shift[event_n][valid_layer_ids]
                # event[valid, 3] -= z_shift[event_n][valid_layer_ids]

            # no box selection, as your data is already restricted to the box.
            inbox_mask = self.box_selection(
                event,
                restrict_x=False,
                restrict_y=False,
                restrict_z=False,  # , box_cut=[-450, 450, -450, 450]
            )
            if inbox_mask.sum() == 0:
                print(f"Event {event_n} has no hits in box, skipping.")
                continue
            event = event[inbox_mask]
            # print(event.shape)
            t = t[inbox_mask]

            # in global coordinates: cut the backscattered hits
            ecal_barrel_inner_radius = 1804.8
            mask = event[:, 1] >= ecal_barrel_inner_radius
            event = event[mask]
            t = t[mask]
            if event.shape[0] == 0:
                print(f"Event {event_n} has no hits after backscatter cut, skipping.")
                continue

            # energy cut at 1e-12 GeV, to remove noise hits
            energy_cut = 1e-12
            mask = event[:, 3] > energy_cut
            event = event[mask]
            t = t[mask]

            if event.shape[0] == 0:
                print(f"Event {event_n} has no hits after energy cut, skipping.")
                continue

            event = event[np.argsort(t)]
            event_list.append(event)
            max_hits = max(max_hits, event.shape[0])

        # pad until max hits
        padded = np.zeros((len(event_list), max_hits, 4))
        for i, event in enumerate(event_list):
            padded[i, : event.shape[0]] = event
        events = padded

        events = np.array(padded)

        if self.metadata.local_xyz_orientaion:
            original_shape = events.shape
            events_flat = events[..., :3].reshape(-1, 3)  # (74*41917, 3)
            events_rotated_flat = self.global_to_local_points(events_flat.T).T
            events_rotated = events_rotated_flat.reshape(original_shape[0], original_shape[1], 3)  # (74, 41917, 3)
            events[..., :3] = events_rotated
        return self.digitize_and_fuzz(events)


_events_global = None
_transform_global = None


def _init_worker(events, transform):
    global _events_global, _transform_global
    _events_global = events
    _transform_global = transform


def _transform_wrapper(args):
    global _events_global, _transform_global
    start, end = args
    try:
        return _transform_global.apply_transformations(start, end, _events_global)
    except Exception as e:
        print(f"Worker error: {e}", flush=True)
        raise


def convert(input_path: str, global_path: str = None, output_folder: str = None):
    if global_path is None:
        # print keys and shapes of the input file
        f = h5py.File(input_path, "r")
        print("Input file keys and shapes:")
        for key in f.keys():
            for subkey in f[key].keys():
                print(f"  {key}/{subkey}: {f[key][subkey].shape}")

        print(f"Reading {input_path}...")
        with h5py.File(input_path, "r") as h5:
            # --- primary info ---
            p_evt = np.asarray(h5["primary"]["event_id"], dtype=np.int32)
            p_mom = np.asarray(h5["primary"]["momentum"], dtype=np.float32)  # (N, 3)
            p_vertex = np.asarray(h5["primary"]["vertex"], dtype=np.float32)  # (N, 3)

            # --- steps info ---
            s_evt = np.asarray(h5["steps"]["event_id"], dtype=np.int32)
            # s_pos = np.asarray(h5["steps"]["position"], dtype=np.float32)  # (M, 3)
            # s_ene = np.asarray(h5["steps"]["energy"], dtype=np.float32)  # (M,)

        unique_ids = np.unique(p_evt)
        n_events = len(unique_ids)
        print(f"Found {n_events} events.")

        # --- compute incident energy per shower from momentum ---
        # E = |p| (massless approximation, valid for photons/electrons at high energy)
        # If you have mass: E = sqrt(mass^2 + |p|^2)
        incident_energies = np.linalg.norm(p_mom, axis=1).astype(np.float32)  # (N,)

        # Build per-event boundaries using searchsorted (assumes s_evt is sorted)
        # boundaries[i] = first step index for event i
        # boundaries[i+1] = first step index for event i+1 (= exclusive end)
        boundaries = np.searchsorted(s_evt, unique_ids)  # (N,)
        boundaries = np.append(boundaries, len(s_evt))  # (N+1,) sentinel at end
        del s_evt  # free ~3 GB immediately

        # --- build events array: (N, max_hits, 4) = x, y, z, energy ---
        # first pass: find max hits per event
        ncpu, batchsize = 4, 8
        pool = mp.Pool(ncpu)

        total_events = len(unique_ids)
        batch_starts = np.arange(0, total_events, batchsize)
        batch_ends = batch_starts + batchsize
        batch_ends[-1] = total_events
        batch_start_end = np.vstack([batch_starts, batch_ends]).T

        args = [(int(s), int(e), boundaries) for s, e in zip(batch_starts, batch_ends)]
        max_hits = pool.starmap(get_max_hits_batch, args)
        max_hits = max(max_hits)
        print(f"Max hits per event: {max_hits}")

        events = np.zeros((n_events, max_hits, 4), dtype=np.float32)
        args = [(input_path, int(s), int(e), boundaries, max_hits) for s, e in zip(batch_starts, batch_ends)]

        print(f"Building event array with shape {events.shape}...")
        with Pool(ncpu) as pool:
            chunks = list(tqdm(pool.imap(build_events_wrapper, args), total=len(args)))

        events = np.vstack(chunks)
        print(f"Final shape: {events.shape}")

        # --- p_global: unit momentum vector ---
        p_norm = p_mom / np.linalg.norm(p_mom, axis=1, keepdims=True)
        p_global = p_norm.astype(np.float32)

        # --- input_p_global: assumed same as p_global ---
        # (if your data has a separate input direction, replace this)
        input_p_global = p_global.copy()

        # --- input_gun_position: primary vertex ---
        input_gun_position = p_vertex.astype(np.float32)

        print("Writing input_global_cc3... in ", os.path.dirname(input_path))
        dict_ = {
            "energy": incident_energies,
            "events": events,
            "p_global": p_global,
            "input_p_global": input_p_global,
            "input_gun_position": input_gun_position,
            "p_mom": p_mom,
        }
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        global_path = input_path.replace(".h5", ".input_global_cc3.h5")
        with h5py.File(global_path, "w") as out:
            for key, value in dict_.items():
                if key != "events":
                    out.create_dataset(key, data=value)

            # Large events array needs to be written in chunks to avoid memory issues
            n_events, max_hits, n_feat = events.shape
            ds = out.create_dataset(
                "events",
                shape=(n_events, max_hits, n_feat),
                dtype=np.float32,
                chunks=(min(512, n_events), max_hits, n_feat),  # chunk per batch
            )
            chunk_size = 512
            for lo in range(0, n_events, chunk_size):
                hi = min(lo + chunk_size, n_events)
                ds[lo:hi] = events[lo:hi]

        print("Done!")
        for key, value in dict_.items():
            print(f"  {key}:             {value.shape}")
        del dict_  # free memory
    else:
        assert os.path.exists(global_path), "Global file must be created first to compute local transformations."
        print(f"Reading global file from {global_path}...")
        f = h5py.File(global_path, "r")
        max_len = f[list(f.keys())[1]].shape[0]
        tot = max_len  # 35000  # for testing, use only a subset of events to speed up
        events = f["events"][:tot]
        incident_energies = f["energy"][:tot]
        p_mom = f["p_mom"][:tot]
        p_global = f["p_global"][:tot]
        input_p_global = f["input_p_global"][:tot]
        # input_gun_position = f["input_gun_position"][:]

    print("Creating CC3 input showers...")
    metadata = Metadata()
    transform = Transform_pointcloud(metadata)
    ncpu, batchsize = 64, 512  # 64, 512

    iE = incident_energies
    mox = p_mom[:, 0]
    moy = p_mom[:, 1]
    moz = p_mom[:, 2]
    moment_angles = transform.get_moment_angles(np.array([mox, moy, moz]).T)

    start_time = time.time()
    print("Creating {}-process pool".format(ncpu))
    pool = mp.Pool(ncpu)

    total_events = len(events[:, :, 0])
    batch_starts = np.arange(0, total_events, batchsize)
    batch_ends = batch_starts + batchsize
    batch_ends[-1] = total_events
    batch_start_end = np.vstack([batch_starts, batch_ends]).T

    print("Applying transformations to incident energy.")
    incident_arguments = [(start, end, iE) for start, end in batch_start_end]
    energies = pool.starmap(incident_energy, incident_arguments)
    energies = np.vstack(energies)

    print("Applying transformations to point clouds.")

    # shm = shared_memory.SharedMemory(create=True, size=events.nbytes)
    # shared_events = np.ndarray(events.shape, dtype=events.dtype, buffer=shm.buf)
    # shared_events[:] = events  # copy once into shared memory
    # pool = mp.Pool(ncpu, initializer=_init_worker, initargs=(shm.name, events.shape, events.dtype, transform))
    # point_clouds = list(tqdm(pool.imap(_transform_wrapper, arguments), total=len(arguments)))
    # pool.close()
    # pool.join()
    # shm.close()
    # shm.unlink()
    bs = 10_000
    pc_list = []
    for i in range(0, total_events, bs):
        start_ = i
        end_ = min(i + bs, total_events)
        tot = end_ - start_
        if tot == 0:
            continue
        print(f"Processing {start_} until {end_} / {total_events}")

        batch_starts = np.arange(0, tot, batchsize)
        batch_ends = np.minimum(batch_starts + batchsize, tot)
        arguments = list(zip(batch_starts, batch_ends))

        _init_worker(events[start_:end_], transform)
        point_clouds = [_transform_wrapper(arg) for arg in tqdm(arguments)]
        pc_list += point_clouds

    # pad until max hits
    max_hits = max(pc.shape[1] for pc in pc_list)
    point_clouds = np.zeros((total_events, max_hits, 4), dtype=np.float32)
    start = 0
    for pc in pc_list:
        actual = pc.shape[0]
        end = min(start + actual, point_clouds.shape[0])
        rows = end - start
        if rows <= 0:
            break  # point_clouds is full
        point_clouds[start:end, : pc.shape[1], :] = pc[:rows]
        start = end

    del pc_list  # free memory
    print(f"Point clouds shape after transformations: {point_clouds.shape}")
    print("Transformations applied. Now sorting by number of points and writing to file.")
    num_points = (events[..., -1] > 0).sum(axis=1)
    if metadata.sort:
        indices = np.argsort(num_points)
        energies = energies[indices]
        events = events[indices]
        for key in moment_angles:
            moment_angles[key] = moment_angles[key][indices]

    energies = energies.astype(np.float32)
    point_clouds = point_clouds.astype(np.float32)
    moment_angles["theta_global"] = moment_angles["theta_global"].astype(np.float32)
    moment_angles["phi_global"] = moment_angles["phi_global"].astype(np.float32)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        out_file = f"{output_folder}/input_cc3.h5"
    else:
        seed = int(input_path.split("/")[-2].split("_")[1])
        algo = input_path.split("/")[-1].split(".")[0].split("compressed_")[-1]
        out_file = f"outputs/cc3input_{algo}/input_cc3_file_{seed}.h5"
        os.makedirs(f"outputs/cc3input_{algo}", exist_ok=True)

    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("energy", data=energies)
        hf.create_dataset("events", data=point_clouds)
        print(point_clouds.shape)
        hf.create_dataset("n_points", data=num_points)
        for key in moment_angles:
            hf.create_dataset(key, data=moment_angles[key])
    hf.close()
    pool.close()
    pool.join()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 2:
        convert(sys.argv[1])
    elif len(sys.argv) == 3:
        convert(sys.argv[1], global_path=sys.argv[2])
    elif len(sys.argv) == 5 and "--pc_save_folder" in sys.argv:
        convert(sys.argv[1], global_path=sys.argv[2], output_folder=sys.argv[4])
    elif len(sys.argv) < 5 and "--pc_save_folder" in sys.argv:
        convert(sys.argv[1], output_folder=sys.argv[3])
    else:
        print("Usage: python convert_to_cc3_format.py input.h5 global_path (optional) --pc_save_folder (optional)")
        sys.exit(1)

""" 
Updated ibllib.pipes.histology.get_brain_regions with this function to update tip size and allow searching for non-root regions within a certain radius of the channel's position.
"""

def get_brain_regions(xyz, channels_positions=None, brain_atlas=None, tip_size_um=175,
                      root_search_radius_um=200, root_search_step_um=None):
    """
    :param xyz: numpy array of 3D coordinates corresponding to a picked track or a trajectory
    the deepest point is assumed to be the tip.
    :param channels_positions:
    :param brain_atlas:
    :return: brain_regions (associated to each channel),
             insertion (object atlas.Insertion, defining 2 points of entries
             (tip and end of probe))
    """
    """
    this is the depth along the probe (from the first point which is the deepest labeled point)
    Due to the blockiness, depths may not be unique along the track so it has to be prepared
    """

    brain_atlas = brain_atlas or atlas.AllenAtlas(10)
    if channels_positions is None:
        geometry = trace_header(version=1)
        channels_positions = np.c_[geometry['x'], geometry['y']]

    xyz = xyz[np.argsort(xyz[:, 2]), :]
    d = atlas.cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
    indsort = np.argsort(d)
    xyz = xyz[indsort, :]
    d = d[indsort]
    iduplicates = np.where(np.diff(d) == 0)[0]
    xyz = np.delete(xyz, iduplicates, axis=0)
    d = np.delete(d, iduplicates, axis=0)

    assert np.all(np.diff(d) > 0), "Depths should be strictly increasing"

    # Get the probe insertion from the coordinates
    insertion = atlas.Insertion.from_track(xyz, brain_atlas)

    # Interpolate channel positions along the probe depth and get brain locations
    xyz_channels = interpolate_along_track(xyz, (channels_positions[:, 1] + tip_size_um) / 1e6)
    probe_vector = xyz[-1] - xyz[0]
    probe_vector = probe_vector / np.linalg.norm(probe_vector)
    def _label_is_invalid(label, cache):
        if label in cache:
            return cache[label]
        if label <= 0:
            cache[label] = True
            return True
        region = brain_atlas.regions.get(np.array([label]))
        acronym = region.acronym[0]
        cosmos = brain_atlas.regions.acronym2acronym(acronym, mapping='Cosmos')
        is_invalid = cosmos.size == 0 or cosmos[0] == 'root' or cosmos[0] == 'void'
        cache[label] = is_invalid
        return is_invalid

    def _search_nearest_non_root(center_xyz, max_radius_um, step_um, cache):
        # Search outward in spherical shells and return the closest non-root label.
        if max_radius_um <= 0:
            return None
        step_um = float(step_um)
        radii_um = np.arange(step_um, float(max_radius_um) + step_um, step_um)
        step_m = step_um / 1e6
        for radius_um in radii_um:
            radius_m = radius_um / 1e6
            offsets = np.arange(-radius_m, radius_m + step_m / 2.0, step_m)
            if offsets.size == 0:
                continue
            dx, dy, dz = np.meshgrid(offsets, offsets, offsets, indexing='ij')
            sphere_mask = (dx ** 2 + dy ** 2 + dz ** 2) <= (radius_m ** 2)
            if not np.any(sphere_mask):
                continue

            candidates = center_xyz + np.c_[dx[sphere_mask], dy[sphere_mask], dz[sphere_mask]]
            candidate_labels = brain_atlas.get_labels(candidates)
            labels, num_occurances = np.unique(candidate_labels, return_counts=True)
            sort_indices = np.argsort(num_occurances)[::-1]
            labels = labels[sort_indices]
            for label in labels:
                if not _label_is_invalid(label, cache):
                    return label
        return None

    labels = brain_atlas.get_labels(xyz_channels).astype(np.int64)
    root_cache = {}
    if root_search_step_um is None:
        root_search_step_um = np.min(np.abs(brain_atlas.bc.dxyz)) * 1e6

    for i, label in enumerate(labels):
        if _label_is_invalid(int(label), root_cache):
            new_label = _search_nearest_non_root(
                xyz_channels[i],
                max_radius_um=root_search_radius_um,
                step_um=root_search_step_um,
                cache=root_cache,
            )
            if new_label is not None:
                labels[i] = new_label

    brain_regions = brain_atlas.regions.get(labels)
    brain_regions['xyz'] = xyz_channels
    brain_regions['lateral'] = channels_positions[:, 0]
    brain_regions['axial'] = channels_positions[:, 1]
    return brain_regions, insertion

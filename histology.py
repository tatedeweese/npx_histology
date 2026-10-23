import shutil
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si
import vedo
from brainrender import Scene, settings
from brainrender.actors import Line
from iblatlas.atlas import AllenAtlas, Insertion
from ibllib.pipes.histology import get_brain_regions
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from tqdm import tqdm

ATLAS_RESOLUTION_UM = 25

# REGION MAPPINGS
_CORTICAL_REGIONS = {
    "ACA": ["ACAd", "ACAv"],
    "ORB": ["ORBl", "ORBm", "ORBvl"],
    "RSP": ["RSPagl", "RSPd", "RSPv"],
    "MO": ["MOp", "MOs"],
    "SS": [
        "SSp-n",
        "SSp-bfd",
        "SSp-ll",
        "SSp-m",
        "SSp-ul",
        "SSp-tr",
        "SSp-un",
        "SSs",
    ],
    "VIS": [
        "VISal",
        "VISam",
        "VISl",
        "VISp",
        "VISpl",
        "VISpm",
        "VISli",
        "VISpor",
        "VISrl",
    ],
    "AUD": ["AUDd", "AUDp", "AUDpo", "AUDv"],
    "Unknown": ["void", "root"],
}
_SUBCORTICAL_REGIONS = {
    "OLF": ["MOB", "AOB", "AON", "TTd", "TTv", "PIR", "NLOT", "TR"],
    "STR": ["CP", "ACB", "FS", "OT", "LSc", "LSr", "LSv", "SF", "SH"],
    "HPF": [
        "CA1",
        "CA2",
        "CA3",
        "DG",
        "FC",
        "IG",
        "ENTl",
        "ENTm",
        "PAR",
        "POST",
        "PRE",
        "SUB",
        "ProS",
    ],
    "DORsm": [
        "VAL",
        "VM",
        "VPL",
        "VPLpc",
        "VPM",
        "VPMpc",
        "PoT",
        "SPF",
        "SPA",
        "PP",
        "MG",
        "LGd",
    ],
    "DORpm": [
        "LP",
        "PO",
        "POL",
        "SGN",
        "Eth",
        "AV",
        "AM",
        "AD",
        "IAM",
        "IAD",
        "LD",
        "IMD",
        "MD",
        "SMT",
        "PR",
        "PVT",
        "PT",
        "RE",
        "Xi",
        "RH",
        "CM",
        "PCN",
        "CL",
        "PF",
        "PIL",
        "RT",
        "IGL",
        "IntG",
        "LGv",
        "SubG",
        "MH",
        "LH",
    ],
    "HY": [
        "SO",
        "PVH",
        "PVa",
        "PVi",
        "ARH",
        "ADP",
        "AVP",
        "AVPV",
        "DMH",
        "MEPO",
        "MPO",
        "PS",
        "PVp",
        "PVpo",
        "SBPV",
        "SCH",
        "SFO",
        "VMPO",
        "VLPO",
        "AHN",
        "LM",
        "MM",
        "SUM",
        "TM",
        "MPN",
        "PMd",
        "PMv",
        "PVHd",
        "VMH",
        "PH",
        "LHA",
        "LPO",
        "PSTN",
        "PeF",
        "RCH",
        "STN",
        "TU",
        "ZI",
    ],
    "AMY": [
        "sAMY",
        "AAA",
        "LA",
        "BLA",
        "BMA",
        "MEA",
        "CEA",
        "COAa",
        "COAp",
        "PA",
        "PAA",
        "IA",
        "HATA",
        "BA",
    ],
    "MB": [
        "SCs",
        "IC",
        "NB",
        "SAG",
        "PBG",
        "SNr",
        "VTA",
        "PN",
        "RR",
        "MRN",
        "SCm",
        "PAG",
        "APN",
        "MPT",
        "NOT",
        "NPC",
        "OP",
        "PPT",
        "RPF",
        "CUN",
        "RN",
        "III",
        "MA3",
        "EW",
        "IV",
        "Pa4",
        "VTN",
        "AT",
        "LT",
        "DT",
        "MT",
        "SNc",
        "PPN",
        "IF",
        "IPN",
        "RL",
        "CLI",
        "DR",
    ],
}
TATE_REGIONS_FINE = {**_CORTICAL_REGIONS}
TATE_REGIONS = {
    # Override ACA with mPFC grouping
    "mPFC": ["PL", "ILA", "ACAd", "ACAv", "DP"],
    **{k: v for k, v in _CORTICAL_REGIONS.items() if k != "ACA"},
    # **_CORTICAL_REGIONS,
    **_SUBCORTICAL_REGIONS,
}
TATE_MPFC_REGIONS = {**_CORTICAL_REGIONS, **_SUBCORTICAL_REGIONS}
_MAPPINGS = {
    "tate": TATE_REGIONS,
    "fine": TATE_REGIONS_FINE,
    "mpfc": TATE_MPFC_REGIONS,
}
REGION_COLORS = {
    "mPFC": "limegreen",
    "MO": "#a4eedb",
    "PL": "#40ec6e",
    "ILA": "#42aa4d",
    "ACA": "#a7f194",
    "ORB": "gold",  # "#24895d",
    "DP": "#028500",
    "TTd": "darkseagreen",  # "#62d09f",
    "OLF": "#9ad2bd",
    "VIS": "royalblue",  # "#08848c"
    "HPF": "mediumpurple",
    "RSP": "teal",  # "#1ba69a",
    "TEa": "turquoise",  # "#15b0b4"
    "CA1": "springgreen",
    "CA3": "mediumseagreen",
    "DG": "mediumspringgreen",
    "ProS": "darkolivegreen",
    "CP": "mediumpurple",
    "SNr": "rebeccapurple",
    "PA": "plum",
    "LP": "tomato",
    "MG": "salmon",
    "SGN": "lightcoral",
    "PIL": "peru",
    "PoT": "chocolate",
    "ZI": "goldenrod",
    "PP": "indianred",
    "STR": "mediumpurple",
    "AMY": "plum",
    "DORsm": "salmon",
    "DORpm": "tomato",
    "HY": "firebrick",
    "MB": "orchid",
}
MPFC_SUBREGIONS = {"PL", "ILA", "ACAd", "ACAv", "DP"}


def get_shank_id(channel_loc: np.ndarray) -> np.ndarray | int:
    """
    Determine shank ID from channel position(s).

    Parameters
    ----------
    channel_loc : np.ndarray
        Channel location(s). Either shape (2,) for single channel
        or (n_channels, 2) for multiple channels.

    Returns
    -------
    np.ndarray or int
        Shank ID(s) (0-3). Returns int for single channel, array for multiple.
    """
    n_positions = channel_loc.shape[0] if len(channel_loc.shape) > 1 else 1

    if n_positions > 1:
        shanks = np.ones(n_positions, dtype=int) * -1
        for i in range(n_positions):
            shanks[i] = get_shank_id(channel_loc[i])
        return shanks

    # Spikeinterface x positions
    SI_SH_MAP = {0: 0, 32: 0, 250: 1, 282: 1, 500: 2, 532: 2, 750: 3, 782: 3}

    # Alternative x positions (e.g., from Kilosort)
    SH_MAP = {27: 0, 59: 0, 277: 1, 309: 1, 527: 2, 559: 2, 777: 3, 809: 3}

    x_pos = channel_loc[0]

    if x_pos in SI_SH_MAP:
        return SI_SH_MAP[x_pos]
    elif x_pos in SH_MAP:
        return SH_MAP[x_pos]
    else:
        raise ValueError(
            f"Unexpected x position {x_pos} for channel location {channel_loc}"
        )


@cache
def _build_reverse_mapping(mapping_name: str) -> dict:
    """Build and cache reverse mapping from acronym to region."""
    reverse = {}
    for region, acronyms in _MAPPINGS[mapping_name].items():
        for acronym in acronyms:
            reverse[acronym] = region
    return reverse


def _get_region(beryl_acronym: str, mapping_name: str) -> str:
    """Map a Beryl/Allen acronym to a Tate region."""
    reverse_mapping = _build_reverse_mapping(mapping_name)
    return reverse_mapping.get(beryl_acronym, beryl_acronym)


def get_tate_regions(beryl_acronym: str) -> str:
    """Map acronym to Tate region (mPFC grouped)."""
    return _get_region(beryl_acronym, "tate")


def get_tate_regions_fine(beryl_acronym: str) -> str:
    """Map acronym to Tate region (minimal grouping)."""
    return _get_region(beryl_acronym, "fine")


def get_tate_mpfc_regions(beryl_acronym: str) -> str:
    """Map acronym to Tate region (mPFC grouped)."""
    return _get_region(beryl_acronym, "mpfc")


def get_beryl_acronyms_for_region(region: str, mapping_name: str = "tate") -> list:
    """Get list of Beryl acronyms for a Tate region."""
    return _MAPPINGS[mapping_name].get(region, [region])


# PROBE/CHANNEL FUNCTIONS
def get_probe_trajectory_dataframe(
    subject,
    probe_region,
    analyzer,
    ba=None,
    histology_base: Path | None = None,
) -> pd.DataFrame:
    """Return one row per probe shank with tip and entry coordinates."""
    if histology_base is None:
        histology_base = Path(r"Z:\Psilocybin\Histology")

    print("Loading Allen atlas for trajectory data...", flush=True)
    if ba is None:
        ba = AllenAtlas(ATLAS_RESOLUTION_UM)
        ba.compute_surface()
    print("Allen atlas loaded.", flush=True)
    trajectories = []

    # for _, row in recording_df.iterrows():
    histology_path = histology_base / subject / "processed"
    mpfc_start_depth = None
    vhpc_start_depth = None
    channel_positions = analyzer.get_channel_locations()
    shanks = np.unique(get_shank_id(channel_positions))
    for shank in shanks:
        traj_file = histology_path / f"trajectory_{probe_region}_sh{shank}.csv"
        if traj_file.exists():
            trajectory = pd.read_csv(traj_file, header=None).values * 10.0
            tip = ba.ccf2xyz(trajectory[0], ccf_order="apdvml")
            entry = ba.ccf2xyz(trajectory[1], ccf_order="apdvml")
        elif subject in ["Madison", "Monroe"]:
            entry = np.array([-0.5, 1.7 + 0.25 * shank, 0.0]) * 1e-3
            if mpfc_start_depth is None:
                ml, ap, dv = entry
                ix = ba.bc.x2i(ml)
                iy = ba.bc.y2i(ap)
                surface_z = ba.top[iy, ix]
                mpfc_start_depth = surface_z
            insertion = Insertion(
                x=entry[0],
                y=entry[1],
                z=mpfc_start_depth,
                phi=90,
                theta=12,
                depth=4e-3,
            )
            entry = insertion.entry
            tip = insertion.tip
        elif subject == "T22" and probe_region == "vHPC":
            entry = np.array([-1.2 - 0.25 * shank, -2.92, 0]) * 1e-3
            if vhpc_start_depth is None:
                ml, ap, dv = entry
                ix = ba.bc.x2i(ml)
                iy = ba.bc.y2i(ap)
                surface_z = ba.top[iy, ix]
                vhpc_start_depth = surface_z
            insertion = Insertion(
                x=entry[0],
                y=entry[1],
                z=vhpc_start_depth,
                phi=0,
                theta=15,
                depth=5.5e-3,
            )
            entry = insertion.entry
            tip = insertion.tip
        else:
            raise ValueError(
                f"No trajectory data found for {subject} / {probe_region} / shank {shank}"
            )
        # convert entry to bregma coordinates

        trajectories.append(
            {
                "subject": subject,
                "probe_region": probe_region,
                "shank": shank,
                "tip": np.asarray(tip).tolist(),
                "entry": np.asarray(entry).tolist(),
            }
        )

    return pd.DataFrame(trajectories)


def visualize_probe_trajectories(
    reocrding_df: pd.DataFrame,
    output_path=None,
    interactive: bool = False,
    include_brain: bool = True,
):
    """Visualize trajectories from get_probe_trajectory_dataframe()."""
    print("\n" + "=" * 80)
    print("PROBE TRAJECTORIES (3D Visualization)")
    print("=" * 80)

    import vedo
    from brainrender import Scene, settings
    from brainrender.actors import Line

    # build trajectory dataframe
    trajectory_df = pd.DataFrame()
    for row in reocrding_df.itertuples(index=False):
        subject = row.subject
        probe_region = row.probe_region
        analyzer_path = Path(row.analyzer_path)
        analyzer = si.load_sorting_analyzer(analyzer_path)
        row_df = get_probe_trajectory_dataframe(subject, probe_region, analyzer)
        trajectory_df = pd.concat([trajectory_df, row_df], ignore_index=True)

    vedo.settings.default_backend = "vtk"
    settings.SHOW_AXES = False

    plotter = vedo.Plotter(offscreen=not interactive)
    print("Creating brainrender scene...", flush=True)
    scene = Scene(
        root=False,
        atlas_name=f"allen_mouse_{ATLAS_RESOLUTION_UM}um" if include_brain else None,
        check_latest=False,
        inset=False,
        plotter=plotter,
    )
    scene.add_brain_region(
        "root",
        alpha=0.15,
        color=[0.8, 0.8, 0.8],
        silhouette=False,
    )

    print("Brainrender scene created.", flush=True)
    ba = AllenAtlas(ATLAS_RESOLUTION_UM)
    colors = {"mPFC": "#FF00F2", "vHPC": "#00E8F5"}

    for _, row in trajectory_df.iterrows():
        probe_xyz = np.asarray([row["tip"], row["entry"]])
        probe_ccf = ba.xyz2ccf(probe_xyz, ccf_order="apdvml")
        probe = Line(
            probe_ccf,
            color=colors.get(row["probe_region"], "#FFFFFF"),
            linewidth=2.5,
            alpha=0.8,
        )
        scene.add(probe)
    scene.plotter.reset_clipping_range()
    alpha = 0.15
    scene.add_brain_region("PL", alpha=alpha, hemisphere="right", silhouette=False)
    scene.add_brain_region("ILA", alpha=alpha, hemisphere="right", silhouette=False)
    scene.add_brain_region("DP", alpha=alpha, hemisphere="right", silhouette=False)
    # scene.add_brain_region("ACAd", alpha=0.1, hemisphere="right")
    # scene.add_brain_region("ACAv", alpha=0.1, hemisphere="right")
    # scene.add_brain_region("MO", alpha=0.1, hemisphere="right")
    # scene.add_brain_region("ORB", alpha=0.1, hemisphere="right")

    scene.add_brain_region(
        "HPF", alpha=alpha, hemisphere="right", silhouette=False, color="mediumpurple"
    )

    print(f"Rendering {len(trajectory_df)} trajectories...", flush=True)
    scene.render(interactive=interactive, axes=0, zoom=1)
    print("Render finished.", flush=True)
    if output_path is None:
        output_path = Path("D:/Psilocybin/Histology/probe_trajectories.png")
    if output_path is not None:
        print(f"Taking screenshot: {output_path}", flush=True)
        scene.screenshot(output_path, format="png", transparent=True)
        scene.screenshot(output_path.with_suffix(".svg"), format="svg")
        print(f"Saved {output_path}")


def unit_count_by_region(recording_df, region_property="tate_region", session=None):
    """Count units by brain region for each recording in the DataFrame."""
    if session is not None and "session" in recording_df.columns:
        recording_df = recording_df[recording_df["session"] == session]

    assert not recording_df.empty

    # split by treatment
    unit_counts = []
    for treatment in recording_df["treatment"].unique():
        treatment_df = recording_df[recording_df["treatment"] == treatment]
        print(f"\nTreatment: {treatment}")
        for _, row in treatment_df.iterrows():
            subject = row["subject"]
            probe_region = row["probe_region"]
            analyzer_path = Path(row["analyzer_path"])
            analyzer = si.load_sorting_analyzer(analyzer_path)
            sorting = analyzer.sorting
            if region_property not in sorting.get_property_keys():
                raise ValueError(
                    f"  {subject} / {probe_region}: {region_property} not found in sorting properties."
                )
            regions = sorting.get_property(region_property)
            unique, counts = np.unique(regions, return_counts=True)
            for region, count in zip(unique, counts):
                unit_counts.append(
                    {
                        "subject": subject,
                        "probe_region": probe_region,
                        "treatment": treatment,
                        "region_property": region_property,
                        "brain_region": region,
                        "unit_count": count,
                    }
                )
    unit_counts_df = pd.DataFrame(unit_counts)
    print("\nUnit counts and subject counts by brain region:")
    print(
        unit_counts_df.groupby(["treatment", "brain_region"])
        .agg(
            unit_count=("unit_count", "sum"),
            subject_count=("subject", "nunique"),
        )
        .reset_index()
    )
    return unit_counts_df


def analyze_unit_counts(
    recording_df,
    region_property="tate_region",
    session=None,
    output_path="D:/Psilocybin/Histology/unit_counts.png",
    min_units=20,
    treatment="psilocybin",
):
    """Plot Allen atlas slices colored by regions containing recorded units."""
    print("\n" + "=" * 80)
    print("ATLAS SLICES (Allen Mouse Brain)")
    print("=" * 80)

    unit_counts_df = unit_count_by_region(
        recording_df, region_property=region_property, session=session
    )
    if unit_counts_df.empty:
        raise ValueError("No unit counts were found for the requested session.")

    psilocybin_df = unit_counts_df[unit_counts_df["treatment"] == treatment]

    region_summary = (
        psilocybin_df.groupby("brain_region")
        .agg(
            unit_count=("unit_count", "sum"),
            subject_count=("subject", "nunique"),
        )
        .sort_values("unit_count", ascending=False)
    )
    region_summary = region_summary[region_summary["unit_count"] >= min_units]

    if region_summary.empty:
        raise ValueError(f"No regions found with at least {min_units} units.")

    # Get the color mappings
    if region_property == "tate_region":
        mapping_name = "tate"
    elif "fine" in region_property:
        mapping_name = "fine"
    elif "mpfc" in region_property:
        mapping_name = "mpfc"
    else:
        raise ValueError(
            f"Unknown region property '{region_property}'. Expected 'tate_region', 'tate_fine_region', or 'tate_mpfc_region'."
        )
    ba = AllenAtlas(res_um=ATLAS_RESOLUTION_UM)
    br = ba.regions
    br.rgb[:] = 255  # Start with white background

    def color_region(acronym, color):
        rgb = np.asarray(to_rgb(color)) * 255
        beryl_acronyms = get_beryl_acronyms_for_region(acronym, mapping_name)
        for beryl_acronym in beryl_acronyms:
            try:
                region_idx = br.acronym2index(beryl_acronym)[1]
            except (IndexError, KeyError):
                continue
            br.rgb[region_idx] = rgb
            region_id = br.id[region_idx]
            for descendant_id in br.descendants(region_id)["id"]:
                descendant_idx = br.id2index(descendant_id)[1]
                br.rgb[descendant_idx] = rgb

    legend_patches = []
    for region, summary in region_summary.iterrows():
        if region == "Unknown":
            continue  # Skip unknown regions
        color = REGION_COLORS.get(region, "lightgray")
        if region == "mPFC":
            for subregion in MPFC_SUBREGIONS:
                color_region(subregion, color)
        else:
            color_region(region, color)
        legend_patches.append(
            mpatches.Patch(
                color=color,
                label=f"{region} (n={summary['subject_count']}, units={summary['unit_count']})",
            )
        )

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    mpfc_ap = 1.78 / 1e3
    vhpc_ap = -2.78 / 1e3
    ba.plot_cslice(mpfc_ap, volume="annotation", ax=axs[0])
    ba.plot_cslice(vhpc_ap, volume="annotation", ax=axs[1])
    ba.plot_cslice(mpfc_ap, volume="boundary", ax=axs[0], alpha=0.7)
    ba.plot_cslice(vhpc_ap, volume="boundary", ax=axs[1], alpha=0.7)

    axs[0].set_title("1.78 mm", fontsize=14)
    axs[1].set_title("-2.78 mm", fontsize=14)
    for ax in axs:
        ax.axis("off")

    fig.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title=f"Regions (≥{min_units} units)",
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()
    plt.close()
    return region_summary


def get_channel_regions(
    analyzer,
    probe_region: str,
    subject: str,
    ba=None,
    tip_size_um: float = 175,
    search_radius_um: float = 200,
    histology_base: Path | None = None,
    plot: bool = False,
):
    """Get brain regions for each channel based on probe trajectory."""
    if ba is None:
        ba = AllenAtlas(ATLAS_RESOLUTION_UM)

    if histology_base is None:
        histology_base = Path(r"Z:\Psilocybin\Histology")

    channel_positions = analyzer.get_channel_locations()
    channel_ids = analyzer.channel_ids
    shanks = get_shank_id(channel_positions)

    # Initialize arrays for channel properties
    n_channels = len(channel_ids)
    channel_allen = np.array(["Unknown"] * n_channels, dtype=object)
    channel_beryl = np.array(["Unknown"] * n_channels, dtype=object)
    channel_tate = np.array(["Unknown"] * n_channels, dtype=object)
    channel_tate_fine = np.array(["Unknown"] * n_channels, dtype=object)
    channel_tate_mpfc = np.array(["Unknown"] * n_channels, dtype=object)
    trajectory_df = get_probe_trajectory_dataframe(
        subject,
        probe_region,
        analyzer,
        ba,
        histology_base=histology_base,
    )

    # Only create scene if plotting
    if plot:
        vedo.settings.default_backend = "vtk"
        settings.SHOW_AXES = False
        scene = Scene(
            title=f"{subject} - {probe_region}",
            atlas_name=f"allen_mouse_{ATLAS_RESOLUTION_UM}um",
            inset=False,
        )

    for sh_id in np.unique(shanks):
        trajectory_rows = trajectory_df[trajectory_df["shank"] == sh_id]
        if trajectory_rows.empty:
            continue
        trajectory_row = trajectory_rows.iloc[0]
        tip_xyz = np.asarray(trajectory_row["tip"])
        entry_xyz = np.asarray(trajectory_row["entry"])

        probe = np.array([tip_xyz, entry_xyz])

        if plot:
            probe_ccf = ba.xyz2ccf(probe, ccf_order="apdvml")
            probe_actor = Line(probe_ccf, linewidth=2, alpha=1)
            scene.add(probe_actor)

        shank_channels = np.where(shanks == sh_id)[0]
        shank_channel_positions = channel_positions[shank_channels]

        brain_regions = get_brain_regions(
            probe,
            shank_channel_positions,
            ba,
            tip_size_um=tip_size_um,
            root_search_radius_um=search_radius_um,
        )

        rids = brain_regions[0].id
        allen_regions = [ba.regions.get(rid).acronym for rid in rids]
        beryl_regions = [
            ba.regions.acronym2acronym(allen_acr, mapping="Beryl")[0]
            for allen_acr in allen_regions
        ]
        tate_regions = [get_tate_regions(beryl_acr) for beryl_acr in beryl_regions]
        tate_fine_regions = [
            get_tate_regions_fine(beryl_acr) for beryl_acr in beryl_regions
        ]
        tate_mpfc_regions = [
            get_tate_mpfc_regions(beryl_acr) for beryl_acr in beryl_regions
        ]
        for i, ch_idx in enumerate(shank_channels):
            channel_allen[ch_idx] = allen_regions[i]
            channel_beryl[ch_idx] = beryl_regions[i]
            channel_tate[ch_idx] = tate_regions[i]
            channel_tate_fine[ch_idx] = tate_fine_regions[i]
            channel_tate_mpfc[ch_idx] = tate_mpfc_regions[i]

    # Only render if plotting
    if plot:
        scene.render(interactive=True, axes=0, zoom=1)
        scene.close()

    return {
        "allen_region": channel_allen,
        "beryl_region": channel_beryl,
        "tate_region": channel_tate,
        "tate_fine_region": channel_tate_fine,
        "tate_mpfc_region": channel_tate_mpfc,
        "shank": shanks,
    }


def add_histology_to_analyzer(
    analyzer,
    subject: str,
    probe_region: str,
    ba=None,
    histology_base: Path | None = None,
    save: bool = True,
    plot: bool = False,
):
    """Add histology-based region properties to a sorting analyzer."""
    unit_ids = analyzer.unit_ids
    n_units = len(unit_ids)
    sorting = analyzer.sorting

    channel_regions = get_channel_regions(
        analyzer,
        probe_region,
        subject,
        ba,
        histology_base=histology_base,
        plot=plot,
    )

    for key, value in channel_regions.items():
        analyzer.recording.set_property(key, value)

    # Initialize unit property arrays
    unit_shank = np.full(n_units, -1, dtype=int)
    unit_allen = np.array(["Unknown"] * n_units, dtype=object)
    unit_beryl = np.array(["Unknown"] * n_units, dtype=object)
    unit_tate = np.array(["Unknown"] * n_units, dtype=object)
    unit_tate_fine = np.array(["Unknown"] * n_units, dtype=object)
    unit_tate_mpfc = np.array(["Unknown"] * n_units, dtype=object)

    extremum_channels = si.get_template_extremum_channel(analyzer)
    channel_id_to_idx = {ch_id: idx for idx, ch_id in enumerate(analyzer.channel_ids)}

    for i, unit_id in enumerate(unit_ids):
        peak_ch_id = extremum_channels[unit_id]
        ch_idx = channel_id_to_idx[peak_ch_id]

        unit_shank[i] = channel_regions["shank"][ch_idx]
        unit_allen[i] = channel_regions["allen_region"][ch_idx]
        unit_beryl[i] = channel_regions["beryl_region"][ch_idx]
        unit_tate[i] = channel_regions["tate_region"][ch_idx]
        unit_tate_fine[i] = channel_regions["tate_fine_region"][ch_idx]
        unit_tate_mpfc[i] = channel_regions["tate_mpfc_region"][ch_idx]

    # Set properties on sorting
    sorting = analyzer.sorting
    sorting.set_property("shank", unit_shank)
    sorting.set_property("allen_region", unit_allen)
    sorting.set_property("beryl_region", unit_beryl)
    sorting.set_property("tate_region", unit_tate)
    sorting.set_property("tate_fine_region", unit_tate_fine)
    sorting.set_property("tate_mpfc_region", unit_tate_mpfc)

    if save:
        sorting.save(folder=analyzer.folder / "sorting", overwrite=True)
        properties_folder = analyzer.folder / "recording_info" / "properties"
        if properties_folder.exists():
            shutil.rmtree(properties_folder)
        analyzer.recording.save_metadata_to_folder(analyzer.folder / "recording_info")

    return analyzer


def add_histology(
    recording_df: pd.DataFrame,
    histology_base: Path | None = None,
    force_recompute: bool = False,
    plot: bool = False,
) -> None:
    """
    Add histology properties to all sorting analyzers in the recording DataFrame.
    Parameters
    ----------
    recording_df : pd.DataFrame
        Recording-level DataFrame from paper1.load_paper1_data()
        Required columns: subject, probe_region, analyzer_path
    histology_base : Path, optional
        Base path for histology data
    force_recompute : bool
        If True, recompute even if properties already exist
    """
    print("Loading Allen Atlas...")
    atlas = None

    for idx, row in tqdm(
        recording_df.iterrows(),
        total=len(recording_df),
        desc="Processing recordings",
    ):
        subject = row["subject"]
        probe_region = row["probe_region"]
        analyzer_path = Path(row["analyzer_path"])

        analyzer = si.load_sorting_analyzer(analyzer_path)

        # Check if already computed
        if not force_recompute:
            existing_props = analyzer.sorting.get_property_keys()
            if "tate_region" in existing_props:
                continue
        if atlas is None:
            atlas = AllenAtlas(ATLAS_RESOLUTION_UM)
            atlas.compute_surface()

        print(f"Processing {subject} / {probe_region}")
        add_histology_to_analyzer(
            analyzer=analyzer,
            subject=subject,
            probe_region=probe_region,
            ba=atlas,
            histology_base=histology_base,
            save=True,
            plot=plot,
        )

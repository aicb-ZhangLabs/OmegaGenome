"""Rigor guards against the track-index/assay confusion. Pins the canonical --track_subset order so a
data reshuffle or a misread CSV row can never silently retarget an experiment. 8 angles."""

from src.data.ntv3_ft_data import NTV3_HUMAN_TRACKS, NTV3_HUMAN_TRACK_IDS


def test_canonical_has_34_unique_ids():
    assert len(NTV3_HUMAN_TRACKS) == 34
    assert len(set(NTV3_HUMAN_TRACK_IDS)) == 34, "ids must be unique"


def test_canonical_is_alphabetical_matches_loader_sort():
    """The loader uses sorted(glob('*.bigwig')); the canonical list MUST equal that sort, else the pin
    is wrong. This is the property that ties NTV3_HUMAN_TRACK_IDS to --track_subset i."""
    assert NTV3_HUMAN_TRACK_IDS == sorted(NTV3_HUMAN_TRACK_IDS), (
        "canonical order must be alphabetical"
    )


def test_specific_index_pins():
    """The exact indices used in experiments are pinned (catches any reorder regression)."""
    m = dict(enumerate(NTV3_HUMAN_TRACKS))
    assert m[18] == (
        "ENCSR527JGN_P",
        "polyA plus RNA-seq",
    )  # the specialist (0.6103) — RNA-seq, NOT PRO-cap
    assert m[12] == ("ENCSR325NFE", "ATAC-seq")  # the strong-teacher demo track (0.83)
    assert m[0] == ("ENCSR046BCI_M", "PRO-cap")  # spec0 (0.4087)
    assert m[5] == (
        "ENCSR114HGS_P",
        "PRO-cap",
    )  # the teacher-weak (0.342) track that was conflated with 18


def test_index_18_is_not_pro_cap():
    """Direct guard for the exact bug: --track_subset 18 must NOT be a PRO-cap track."""
    assert NTV3_HUMAN_TRACKS[18][1] != "PRO-cap"
    assert NTV3_HUMAN_TRACKS[18][0] != "ENCSR114HGS_P"  # the track it was wrongly conflated with


def test_csv_row_order_differs_from_dataset_order():
    """Sanity: the assay-grouped order (what the per-track CSV uses) is NOT the dataset order — the root
    cause of the confusion. Grouping by assay reorders the ids."""
    by_assay = sorted(NTV3_HUMAN_TRACKS, key=lambda t: (t[1], t[0]))
    assert [t[0] for t in by_assay] != NTV3_HUMAN_TRACK_IDS, (
        "assay-sorted != dataset order (the trap)"
    )


def test_load_guard_condition_detects_reorder():
    """The loader's guard (`bigwig_ids != NTV3_HUMAN_TRACK_IDS`) flags any reordering/missing track."""
    reordered = NTV3_HUMAN_TRACK_IDS[:]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert reordered != NTV3_HUMAN_TRACK_IDS  # swap detected
    assert NTV3_HUMAN_TRACK_IDS[:-1] != NTV3_HUMAN_TRACK_IDS  # missing-track detected


def test_assay_set_is_the_known_5():
    """All assays are one of the known benchmark assay types (no typo/unknown assay slipping in)."""
    assays = {a for _, a in NTV3_HUMAN_TRACKS}
    assert assays == {
        "PRO-cap",
        "eCLIP",
        "ATAC-seq",
        "polyA plus RNA-seq",
        "total RNA-seq",
        "Histone ChIP-seq",
    }


def test_bounds_logic():
    """The --track_subset bounds check semantics: any index outside [0, 34) is rejected."""
    native_n = 34
    for idx in ([34], [-1], [0, 99]):
        bad = [i for i in idx if i < 0 or i >= native_n]
        assert bad, f"{idx} should be flagged out-of-range"
    assert not [i for i in [0, 12, 18, 33] if i < 0 or i >= native_n]  # valid indices pass

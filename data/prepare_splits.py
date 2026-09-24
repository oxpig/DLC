"""Builds the per-structure feature files used throughout this repo, for
the two published PDB(+chain) code lists in this directory
(headline_34_pdb_codes.txt: paired heavy+light Fv structures;
nanobody_100_pdb_codes.txt: single-domain heavy-only structures), from a
local copy of SAbDab2's public ML-data release.

SAbDab2 (https://sabdab2.opig.stats.ox.ac.uk) publishes train/test splits
(ab_split.csv, ab_split_sd.csv) alongside the mmCIF structure files they
reference - download that release yourself and pass its directory as
--sabdab2-dir (it should contain ab_split.csv, ab_split_sd.csv, and one
.cif per structure instance, named pdb_<PDB_ID>_<Hchain>_<Lchain>.cif with
'+' for a missing chain). No automated single-file download API is used
here since SAbDab2 doesn't publish one - the release is a single archive.

SAbDab2's antibody chains are already IMGT-numbered (a residue's own
auth_seq_id is its IMGT number), so CDR/framework boundaries are read off
directly from residue numbering - no external numbering tool (e.g. ANARCI)
is needed. region_numeric follows AlphaFold2/OpenFold-style integer coding:
0-2=heavy CDR1-3, 6-9=heavy FW1-4, 3-5=light CDR1-3, 10-13=light FW1-4.

Requires: gemmi (mmCIF parsing), and the public `openfold` package (for
its atom14/frame data_transforms - already a dependency of models/openfold.py).
"""
import argparse
import csv
from pathlib import Path

import gemmi
import torch
from openfold.data import data_transforms
from openfold.np import residue_constants

HERE = Path(__file__).parent
COMP_ID_REMAP = {"MSE": "MET"}  # selenomethionine, a common crystallographic Met substitute


def region_numeric(imgt_num: int, is_heavy: bool) -> int:
    """IMGT V-domain region boundaries (inclusive) - identical for heavy
    and light chains by IMGT's own design, offset by chain in the returned
    code so heavy/light CDRs/frameworks never collide."""
    if imgt_num <= 26:
        code = 6  # fw1
    elif imgt_num <= 38:
        code = 0  # cdr1
    elif imgt_num <= 55:
        code = 7  # fw2
    elif imgt_num <= 65:
        code = 1  # cdr2
    elif imgt_num <= 104:
        code = 8  # fw3
    elif imgt_num <= 117:
        code = 2  # cdr3
    else:
        code = 9  # fw4, including any C-terminal overhang past 128
    return code if is_heavy else code + 3 if code < 6 else code + 7


def extract_chain(cif_path: Path, chain_id: str, is_heavy: bool):
    """Reads one chain out of an mmCIF file into per-residue tensors."""
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    model = structure[0]
    if chain_id not in [c.name for c in model]:
        return None
    chain = model[chain_id]

    seq_mask, all_atom_positions, all_atom_mask = [], [], []
    region_codes, sequence_3 = [], []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is None or not info.is_amino_acid():
            continue
        comp_id = COMP_ID_REMAP.get(residue.name, residue.name)
        sequence_3.append(comp_id)
        region_codes.append(region_numeric(residue.seqid.num, is_heavy))

        positions = torch.zeros((37, 3))
        mask = torch.zeros((37,))
        for atom in residue:
            if atom.name in residue_constants.atom_order:
                idx = residue_constants.atom_order[atom.name]
                positions[idx] = torch.tensor([atom.pos.x, atom.pos.y, atom.pos.z])
                mask[idx] = 1.0
        all_atom_positions.append(positions)
        all_atom_mask.append(mask)
        seq_mask.append(int(comp_id in residue_constants.restype_3to1))

    if not sequence_3:
        return None
    return {
        "seq_mask": torch.tensor(seq_mask),
        "all_atom_positions": torch.stack(all_atom_positions),
        "all_atom_mask": torch.stack(all_atom_mask),
        "region_numeric": torch.tensor(region_codes),
        "sequence_3_letters": sequence_3,
        "is_heavy": torch.full((len(sequence_3),), float(is_heavy)),
    }


def concat_chains(chain_dicts):
    out = {}
    for key in chain_dicts[0]:
        if key == "sequence_3_letters":
            out[key] = sum((c[key] for c in chain_dicts), [])
        else:
            out[key] = torch.cat([c[key] for c in chain_dicts], dim=0)
    out["residue_index"] = torch.arange(len(out["sequence_3_letters"]))
    return out


def add_sequence_and_aatype(item: dict) -> dict:
    sequence = "".join(residue_constants.restype_3to1.get(aa, "X") for aa in item["sequence_3_letters"])
    item["sequence"] = sequence
    item["aatype"] = torch.tensor([residue_constants.restype_order_with_x[aa] for aa in sequence])
    return item


def add_atom14_and_frames(item: dict) -> dict:
    # residues without structural information are marked unknown so the
    # atom14 masks/positions/frames openfold derives come out correctly;
    # the real aatype is restored afterwards.
    real_aatype = item["aatype"].clone()
    item["aatype"][~item["seq_mask"].bool()] = residue_constants.restype_order_with_x["X"]

    for transform in (data_transforms.make_atom14_masks, data_transforms.make_atom14_positions):
        item = transform(item)

    item["all_atom_positions"] = item["all_atom_positions"].double()
    for transform in (
        data_transforms.atom37_to_frames,
        data_transforms.atom37_to_torsion_angles(""),
        data_transforms.make_pseudo_beta(""),
        data_transforms.get_backbone_frames,
        data_transforms.get_chi_angles,
    ):
        item = transform(item)

    for key in ("all_atom_positions", "atom14_gt_positions", "torsion_angles_sin_cos",
                "alt_torsion_angles_sin_cos", "pseudo_beta"):
        item[key] = item[key].float()
    item["aatype"] = real_aatype
    return item


def load_split_lookup(sabdab2_dir: Path):
    lookup = {}
    for csv_name in ("ab_split.csv", "ab_split_sd.csv"):
        path = sabdab2_dir / csv_name
        if not path.is_file():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                lookup[row["PDB_ID"]] = row
    return lookup


def build_paired(code: str, sabdab2_dir: Path, split_lookup: dict):
    pdb_id = f"pdb_{code.zfill(8)}"
    row = split_lookup.get(pdb_id)
    if row is None:
        return None, None
    instance = row["INSTANCE"]
    cif_path = sabdab2_dir / f"{instance}.cif"
    if not cif_path.is_file():
        return None, None
    heavy = extract_chain(cif_path, row["Hchain"], is_heavy=True)
    light = extract_chain(cif_path, row["Lchain"], is_heavy=False) if row["Lchain"] != "+" else None
    if heavy is None:
        return None, None
    chains = [heavy] + ([light] if light is not None else [])
    return instance, concat_chains(chains)


def build_nanobody(name: str, sabdab2_dir: Path):
    # name has the form pdb_<PDB_ID>_<Hchain>_+
    parts = name.split("_")
    hchain = parts[-2]
    cif_path = sabdab2_dir / f"{name}.cif"
    if not cif_path.is_file():
        return None
    heavy = extract_chain(cif_path, hchain, is_heavy=True)
    if heavy is None:
        return None
    return concat_chains([heavy])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sabdab2-dir", required=True, type=Path,
                         help="Local directory with SAbDab2's ab_split.csv/ab_split_sd.csv and .cif files")
    parser.add_argument("--out-dir", default=HERE / "structures", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_lookup = load_split_lookup(args.sabdab2_dir)

    with open(HERE / "headline_34_pdb_codes.txt") as f:
        headline_codes = [line.strip() for line in f if line.strip()]
    with open(HERE / "nanobody_100_pdb_codes.txt") as f:
        nanobody_names = [line.strip() for line in f if line.strip()]

    n_ok, n_failed = 0, 0
    for code in headline_codes:
        instance, item = build_paired(code, args.sabdab2_dir, split_lookup)
        if item is None:
            print(f"  [headline/{code}] FAILED (not found in split CSV or missing .cif)")
            n_failed += 1
            continue
        item = add_sequence_and_aatype(item)
        item = add_atom14_and_frames(item)
        item["structure"] = instance
        torch.save(item, args.out_dir / f"{instance}.pt")
        n_ok += 1

    for name in nanobody_names:
        item = build_nanobody(name, args.sabdab2_dir)
        if item is None:
            print(f"  [nanobody/{name}] FAILED (missing .cif or chain)")
            n_failed += 1
            continue
        item = add_sequence_and_aatype(item)
        item = add_atom14_and_frames(item)
        item["structure"] = name
        torch.save(item, args.out_dir / f"{name}.pt")
        n_ok += 1

    print(f"\n{n_ok} structures written to {args.out_dir}, {n_failed} failed")


if __name__ == "__main__":
    main()

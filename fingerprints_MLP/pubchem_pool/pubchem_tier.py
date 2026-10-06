"""Read the PubChem tier of ahmedberatozer/casmi26-pubchem-tier (in external/pubchem_tier/).

The tier is three numpy arrays, one entry per structure, sorted by mass:
- pc_mass.npy: float64 (n,), monoisotopic mass
- pc_off.npy: int64 (n + 1,), structure i's SMILES is bytes off[i]:off[i + 1] of the SMILES blob
- pc_smiles.npy: uint8 blob of all SMILES (ASCII), 5.5 GB

The external disk is FAT32 (no file above 4 GB), so pc_smiles.npy is stored as consecutive parts
pc_smiles.npy.part_aa, _ab, ... (their concatenation is the original .npy file, header included).
"""

from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
TIER_DIR = PROJECT_DIR / "external" / "pubchem_tier"
NPY_HEADER_BYTES = 128  # size of pc_smiles.npy's header: '|u1', shape (5520120062,)


class PubChemTier:
    def __init__(self, tier_dir: Path = TIER_DIR):
        self.masses = np.load(tier_dir / "pc_mass.npy", mmap_mode="r")
        self.offsets = np.load(tier_dir / "pc_off.npy", mmap_mode="r")

        # the parts, memory-mapped, with the position of each one's first byte in the original file
        part_paths = sorted(tier_dir.glob("pc_smiles.npy.part_*"))
        self.parts = [np.memmap(path, dtype=np.uint8, mode="r") for path in part_paths]
        self.part_starts = np.cumsum([0] + [len(part) for part in self.parts[:-1]])

    def __len__(self) -> int:
        return len(self.masses)

    def read_bytes(self, start: int, end: int) -> bytes:
        """Bytes start:end of the SMILES blob (positions after the .npy header), possibly spanning two parts."""
        file_start = start + NPY_HEADER_BYTES
        file_end = end + NPY_HEADER_BYTES
        chunks = []
        for part, part_start in zip(self.parts, self.part_starts):
            part_end = part_start + len(part)
            if file_end <= part_start or file_start >= part_end:
                continue  # no overlap with this part
            first = max(file_start, part_start) - part_start
            last = min(file_end, part_end) - part_start
            chunks.append(part[first:last].tobytes())
        return b"".join(chunks)

    def smiles(self, start: int, end: int) -> list[str]:
        """SMILES of structures start:end (one read for the whole range)."""
        offsets = np.asarray(self.offsets[start:end + 1])
        blob = self.read_bytes(int(offsets[0]), int(offsets[-1]))
        relative = offsets - offsets[0]
        return [blob[relative[i]:relative[i + 1]].decode("ascii") for i in range(end - start)]

    def mass_window(self, neutral_mass: float, ppm: float) -> tuple[int, int]:
        """(start, end) positions of the structures within ppm of neutral_mass."""
        tolerance = neutral_mass * ppm * 1e-6
        start = int(np.searchsorted(self.masses, neutral_mass - tolerance, side="left"))
        end = int(np.searchsorted(self.masses, neutral_mass + tolerance, side="right"))
        return start, end

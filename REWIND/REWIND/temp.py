import bw2data as bd
import bw2io as bi
from pathlib import Path

bd.projects.set_current('wimby')

# --- Biosphere ---
if 'biosphere3' not in bd.databases:
    try:
        bi.bw2setup()
    except ValueError as e:
        if 'biosphere3' not in bd.databases:
            raise
        # bw2data 4.7 / bw2io 0.9.17 mismatch: biosphere3 was created but default
        # LCIA methods failed. Non-fatal, fixed below.
        print(f"Note: bw2setup LCIA step failed ({e.__class__.__name__}). Will fix below.")
    print("Biosphere set up.")

# --- LCIA methods ---
# bw2data 4.7's Method.write() rejects list-format keys from bw2io 0.9.17.
# Patch Method.write to convert lists to tuples before the inner check runs.
if len(bd.methods) < 100:
    from bw2data.method import Method

    _orig_write = Method.write

    def _patched_write(self, data, process=True):
        fixed = [
            ((tuple(line[0]),) + tuple(line[1:])) if isinstance(line[0], list) else line
            for line in data
        ]
        return _orig_write(self, fixed, process=process)

    Method.write = _patched_write
    bi.create_default_lcia_methods(overwrite=True)
    Method.write = _orig_write  # restore

    print(f"LCIA methods installed: {len(bd.methods)}")
    ef_count = len([m for m in bd.methods if 'EF v3.1' in str(m)])
    print(f"EF v3.1 methods: {ef_count}")

# --- Ecoinvent ---
ei_path = Path("/home/elie/Desktop/masters/EVERGI/ReWind/REWIND/REWIND/data/datasets/datasets")

if 'ecoinvent-391-cutoff' not in bd.databases:
    importer = bi.SingleOutputEcospold2Importer(ei_path, 'ecoinvent-391-cutoff', use_mp=False)
    importer.apply_strategies()
    importer.statistics()
    importer.write_database()
    print("Ecoinvent imported successfully.")
else:
    print("Ecoinvent already imported, nothing to do.")

print("\nSetup complete. Databases:", list(bd.databases))
print(f"Total LCIA methods: {len(bd.methods)}")

# data

Output directory for the demos. Each demo writes to its own subfolder:

| Subfolder | Written by | Contents |
|---|---|---|
| `uncertain_assembly_sampling/` | `ur_uncertain_assembly_sampling` | per-run CSV logs (`uncertain_assembly_log_<timestamp>.csv`) |
| `cable_pick_place/` | `ur_cable_pick_place_demo` | per-run scan overlays (`<timestamp>/view_NN.png`, SAM3 predictions on the captured frame) |

Paths are resolved from the working directory you launch from (default `data/...`); run the demos
from the workspace root (`/abhay_ws/ur-assembly`) so they land here. Override per demo via the
config (`csv_path` / `data_dir`).

These are run artifacts — you'll probably want to `.gitignore` the contents (keep this README and
the `.gitkeep` files).

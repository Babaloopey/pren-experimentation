# PREN1 ROS 2 demo: agent notes

A ROS 2 Jazzy digital twin of the PREN1 dog robot (HSLU HS26). Lorenz built it to show his team
what ROS gets them. `README.md` holds architecture, nodes, topics and scripts; read it before changing anything.
For general ROS 2 / RViz / tf2 know-how, load the `ros2-knowledge` skill.

## Project context (Obsidian vault)

Vault: `~/Documents/91_Obsidian/the second brain/` (follow its own `CLAUDE.md` when writing there)

- `00 MOCs/PREN MOC.md`: module hub
- `01 Notes/PREN1 Anforderungsliste.md`: requirement list, the source of truth for what the robot must do
- `01 Notes/PREN1 Recherche Software & Erkennung.md`: the ROS evaluation this demo backs up
- `03 Attachments/pren1_aufgabenstellung.pdf`: official task (field layout, competition flow)

## Invariant

Robot nodes (`ekf`, `field`, `dog_brain`, `search_node`, `cmd_mux`) see only what real hardware will
provide. Ground truth (`/sim/*`, frames `world` / `base_truth` / `camera_truth`) belongs to sim-only
nodes and RViz's "Real world (sim)" group. A new feature keeps that split, so the robot nodes still run unchanged on the Pi.

## Working here

- Every shell sources `run/env.sh` first: it switches the AIR_Week robot settings in `~/.bashrc`
  (discovery server, domain 42) over to local-only (domain 17).
- After editing existing Python, restarting is enough (`--symlink-install`). Rebuild with `run/build.sh` after adding a node, launch arg or config file.
- Herdr: tab `pren-ros-demo` with panes `sim` / `keyboard` / `monitor` / `shell` (recreate: `run/herdr_up.sh`).
  Restart the sim with Ctrl-C in `sim`, then `run/sim.sh`. Pane ids are session-local, so look them up again each session.
- Headless tests: `rviz:=false`, a unique `ROS_DOMAIN_ID` per test, and a small rclpy driver that waits for
  `get_subscription_count() > 0` before publishing. `ros2 topic pub --once` loses messages to discovery.
  Finish with `run/stop.sh`: crashed or backgrounded launches leave orphaned nodes behind.
- RViz text: `marker()` in `common.py` turns spaces into `_` (rviz2 renders spaces very wide). Create markers through it.

## State (2026-10-01)

Done and committed: full mission (SUCH → sweep → touch → report → return → sit), voice/gesture commands
via an RViz button panel + keyboard, EKF fusing wheel odom + IMU, real-world vs. belief view, `/bumper`,
"Zurueck zum Start", `stop.sh`. It hasn't been shown to the team yet.

Open ideas (not requested yet): an `interfaces` package to replace the JSON-in-String `/dog/command`; the
serial/micro-ROS bridge to the MCU; a real vision node replacing `bone_detector`.

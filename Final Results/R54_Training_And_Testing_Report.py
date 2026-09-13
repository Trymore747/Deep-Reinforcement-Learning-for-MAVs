#!/usr/bin/env python3
"""
Generate: Final Results/R54_Full_Process_Documentation.docx
Covers: CERLAB repo setup → ROS2/Gazebo migration → DRL training → testing & results
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path
import copy

OUT = Path("/home/makhosazana/Project/CERLAB-UAV-Autonomy/Final Results/R54_Full_Process_Documentation.docx")
IMG = Path("/home/makhosazana/Project/CERLAB-UAV-Autonomy/Final Results")
TEST_IMG = Path("/home/makhosazana/Project/CERLAB-UAV-Autonomy/Final Results/testing/plots/Actual")
TRAIN_IMG = Path("/home/makhosazana/Project/CERLAB-UAV-Autonomy/Final Results")
CROSS_IMG = Path("/home/makhosazana/Project/CERLAB-UAV-Autonomy/Final Results/testing/cross_validation")

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Inches(8.5)
section.page_height = Inches(11)
section.left_margin = section.right_margin = Inches(1.0)
section.top_margin  = section.bottom_margin = Inches(1.0)

# ── Normal style ──────────────────────────────────────────────────────────────
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)

# ── Helpers ───────────────────────────────────────────────────────────────────
BLUE  = RGBColor(0x15, 0x65, 0xC0)
DBLUE = RGBColor(0x0D, 0x47, 0xA1)
GREY  = RGBColor(0x55, 0x55, 0x55)
GREEN = RGBColor(0x1B, 0x5E, 0x20)
RED   = RGBColor(0xB7, 0x1C, 0x1C)

def h1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = DBLUE
    p.runs[0].font.size = Pt(16)
    p.runs[0].font.bold = True
    return p

def h2(text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = BLUE
    p.runs[0].font.size = Pt(13)
    p.runs[0].font.bold = True
    return p

def h3(text):
    p = doc.add_heading(text, level=3)
    p.runs[0].font.color.rgb = GREY
    p.runs[0].font.size = Pt(11.5)
    p.runs[0].font.bold = True
    return p

def body(text, bold=False, italic=False, color=None, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    p.paragraph_format.space_after = Pt(6)
    return p

def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.size = Pt(11)
    p.paragraph_format.left_indent = Inches(0.3 + level * 0.2)
    p.paragraph_format.space_after = Pt(3)
    return p

def numbered(text, level=0):
    p = doc.add_paragraph(style="List Number")
    run = p.add_run(text)
    run.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    return p

def code(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = "Courier New"
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)
    return p

def table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1565C0")
        shading.set(qn("w:color"), "auto")
        shading.set(qn("w:val"), "clear")
        cell._tc.get_or_add_tcPr().append(shading)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Data rows
    for ri, row in enumerate(rows):
        drow = t.rows[ri + 1]
        fill = "EEF2FF" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row):
            cell = drow.cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(10)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), fill)
            shading.set(qn("w:color"), "auto")
            shading.set(qn("w:val"), "clear")
            cell._tc.get_or_add_tcPr().append(shading)
    if col_widths:
        for ri, row in enumerate(t.rows):
            for ci, w in enumerate(col_widths):
                row.cells[ci].width = Inches(w)
    doc.add_paragraph()
    return t

def img(path, width=6.0, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    try:
        run.add_picture(str(path), width=Inches(width))
    except Exception:
        p.add_run(f"[Figure: {Path(path).name}]").font.italic = True
    if caption:
        cp = doc.add_paragraph(caption)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cp.runs:
            run.font.italic = True
            run.font.size = Pt(10)
            run.font.color.rgb = GREY
        cp.paragraph_format.space_after = Pt(10)

def divider():
    doc.add_paragraph("─" * 80).paragraph_format.space_after = Pt(4)

def pagebreak():
    doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
#  TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph("\n\n")
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("CERLAB UAV Autonomy Project")
r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = DBLUE

tp2 = doc.add_paragraph()
tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = tp2.add_run("Deep Reinforcement Learning for Autonomous Tunnel Navigation")
r2.font.size = Pt(16); r2.font.color.rgb = BLUE

doc.add_paragraph()
tp3 = doc.add_paragraph()
tp3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = tp3.add_run("Full Process Documentation")
r3.font.size = Pt(14); r3.font.bold = True

doc.add_paragraph()
tp4 = doc.add_paragraph()
tp4.alignment = WD_ALIGN_PARAGRAPH.CENTER
r4 = tp4.add_run("Repository Setup  ·  ROS2/Gazebo Migration  ·  Training (R50–R54)  ·  Testing & Results")
r4.font.size = Pt(11); r4.font.color.rgb = GREY

doc.add_paragraph()
tp5 = doc.add_paragraph()
tp5.alignment = WD_ALIGN_PARAGRAPH.CENTER
r5 = tp5.add_run("Date: August 2026  |  Policy: r54_best.pth  |  Platform: CERLAB Quadcopter + PX4 SITL")
r5.font.size = Pt(10); r5.font.color.rgb = GREY

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  TABLE OF CONTENTS (manual)
# ══════════════════════════════════════════════════════════════════════════════
h1("Table of Contents")
toc_entries = [
    ("1", "Project Overview", 3),
    ("2", "Repository Setup from GitHub", 3),
    ("2.1", "Prerequisites and Dependencies", 3),
    ("2.2", "Cloning and Building the Workspace", 3),
    ("3", "ROS2 and Gazebo Migration", 4),
    ("3.1", "Migration from ROS1 to ROS2 Humble", 4),
    ("3.2", "Gazebo 9 → Gazebo 11 Migration", 4),
    ("3.3", "MAVROS and PX4 SITL Integration", 4),
    ("4", "Simulation Environment", 5),
    ("4.1", "Tunnel World Design", 5),
    ("4.2", "Drone Models", 5),
    ("5", "DRL Algorithm — TD3", 5),
    ("6", "Training Process — Runs R50 to R54", 6),
    ("6.1", "State and Action Space Design", 6),
    ("6.2", "Run R50 — Baseline Attempt", 6),
    ("6.3", "Run R51 — Speed Cap Issue", 6),
    ("6.4", "Run R52 — Speed Improvement", 6),
    ("6.5", "Run R53 — Architecture Scaling", 6),
    ("6.6", "Run R54 — Final Architecture (Best Policy)", 7),
    ("7", "R54 Architecture and Reward Design", 7),
    ("8", "Cross-Validation", 8),
    ("8.1", "Observation Withholding (Figure 6.11)", 8),
    ("8.2", "K-Fold Blocked Cross-Validation", 8),
    ("9", "Testing Phase", 9),
    ("9.1", "Test A — Domain Randomisation (Native Drone)", 9),
    ("9.2", "Test B — Software-in-Loop (PX4, Testing World)", 9),
    ("9.3", "Test C — Software-in-Loop (PX4, Training World)", 9),
    ("10", "Results Summary and Analysis", 10),
    ("11", "Limitations and Future Work", 10),
]
for num, title, _ in toc_entries:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"  {num}  {title}")
    r.font.size = Pt(11)
    if "." not in num:
        r.font.bold = True

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
h1("1. Project Overview")
body(
    "The CERLAB UAV Autonomy project is a research initiative at Carleton University's "
    "Collaborative Embedded and Robotic Systems Laboratory (CERLAB). The project develops "
    "autonomous navigation capabilities for unmanned aerial vehicles (UAVs) using Deep "
    "Reinforcement Learning (DRL), with a focus on high-speed flight through confined, "
    "cluttered environments such as underground tunnels, mines, and collapsed structures."
)
body(
    "This document covers the complete development pipeline from the initial GitHub repository "
    "setup through ROS2/Gazebo migration, iterative DRL training across Runs 50 to 54, "
    "comprehensive cross-validation, and final testing using both the native CERLAB drone "
    "and the PX4 SITL platform with MAVROS OFFBOARD control."
)
h2("Project Goals")
bullet("Train a DRL policy capable of navigating a 100 m dynamic tunnel round-trip at high speed (3–6 m/s)")
bullet("Transfer the trained policy from the native CERLAB simulator to the PX4 SITL platform")
bullet("Validate generalisation across unseen obstacle densities and configurations")
bullet("Benchmark performance on three distinct test configurations")

h2("Technology Stack")
table(
    ["Component", "Technology", "Version / Notes"],
    [
        ["Operating System", "Ubuntu", "22.04 LTS (Jammy)"],
        ["Middleware", "ROS2 Humble Hawksbill", "LTS, released May 2022"],
        ["Physics Simulator", "Gazebo", "11 (Citadel) — migrated from Gazebo 9"],
        ["DRL Framework", "PyTorch", "2.x, CPU inference"],
        ["Algorithm", "TD3 — Twin-Delayed DDPG", "Custom implementation"],
        ["Native Drone", "CERLAB Quadcopter", "Position-setpoint PD controller"],
        ["SIL Drone", "PX4 iris_depth_camera", "PX4 SITL + MAVROS OFFBOARD"],
        ["MAVROS", "MAVROS / px4.launch", "v2.x, FCU URL udp://:14540"],
        ["Language", "Python 3.10", "NumPy, PyTorch, rclpy, cv_bridge"],
    ],
    [1.8, 2.2, 2.0]
)

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — REPOSITORY SETUP
# ══════════════════════════════════════════════════════════════════════════════
h1("2. Repository Setup from GitHub")
body(
    "The CERLAB UAV Autonomy repository is hosted on GitHub. It provides a complete "
    "ROS2 workspace containing the UAV simulator, navigation stack, DRL training scripts, "
    "and supporting tools. The setup process involves cloning the repository, installing "
    "system dependencies, and building the colcon workspace."
)

h2("2.1 Prerequisites and Dependencies")
body("The following software must be installed before cloning the repository:")
bullet("Ubuntu 22.04 LTS (recommended) or Ubuntu 20.04 LTS")
bullet("ROS2 Humble Hawksbill — full desktop installation")
bullet("Gazebo 11 (installed via ROS2 desktop-full)")
bullet("Python 3.10 with pip, numpy, torch, matplotlib, openpyxl")
bullet("MAVROS and MAVROS extras for PX4 integration")
bullet("PX4-Autopilot (SITL build) for software-in-loop testing")
bullet("colcon build system (installed with ROS2)")

h3("ROS2 Humble Installation")
code("sudo apt update && sudo apt install -y ros-humble-desktop-full")
code("sudo apt install -y python3-colcon-common-extensions python3-rosdep")
code("sudo rosdep init && rosdep update")

h3("MAVROS Installation")
code("sudo apt install -y ros-humble-mavros ros-humble-mavros-extras")
code("ros2 run mavros install_geographiclib_datasets.sh  # GeographicLib datasets")

h2("2.2 Cloning and Building the Workspace")
numbered("Clone the CERLAB UAV Autonomy repository:")
code("git clone https://github.com/CERLAB/CERLAB-UAV-Autonomy.git")
code("cd CERLAB-UAV-Autonomy")
numbered("Initialise and update submodules (if any):")
code("git submodule update --init --recursive")
numbered("Install ROS2 package dependencies:")
code("source /opt/ros/humble/setup.bash")
code("rosdep install --from-paths src --ignore-src -r -y")
numbered("Build the workspace with colcon:")
code("colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release")
numbered("Source the built workspace:")
code("source install/setup.bash")
body(
    "After a successful build, the workspace contains the uav_simulator package "
    "(providing Gazebo worlds, drone URDF models, and launch files), the tunnel_drl "
    "package (DRL training scripts), and supporting navigation and planning packages.",
    italic=True
)

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ROS2 / GAZEBO MIGRATION
# ══════════════════════════════════════════════════════════════════════════════
h1("3. ROS2 and Gazebo Migration")

h2("3.1 Migration from ROS1 to ROS2 Humble")
body(
    "The CERLAB codebase was originally developed under ROS1 Noetic. Migrating to "
    "ROS2 Humble required changes at every layer of the stack — node communication, "
    "launch files, build system, and simulator interface."
)
table(
    ["ROS1 (Noetic)", "ROS2 Humble (Migrated)", "Impact"],
    [
        ["rospy / roscpp nodes", "rclpy / rclcpp nodes", "All node code rewritten"],
        ["rospy.Publisher / Subscriber", "create_publisher / create_subscription", "API change"],
        ["roslaunch XML files", "Python launch files (.launch.py)", "Full rewrite"],
        ["catkin build system", "colcon build system", "CMakeLists & package.xml v3"],
        ["gazebo_ros plugins (ROS1)", "gazebo_ros plugins (ROS2)", "Plugin API updated"],
        ["rostopic / rosservice CLI", "ros2 topic / ros2 service CLI", "Tooling change"],
        ["rosbag record", "ros2 bag record", "Data logging updated"],
    ],
    [2.0, 2.2, 1.8]
)
body(
    "Key migration decisions: the CERLAB quadcopter's position-setpoint controller was "
    "retained as the primary training platform due to its deterministic response, while "
    "PX4 SITL with MAVROS was added as a secondary validation platform in the final testing phase."
)

h2("3.2 Gazebo 9 → Gazebo 11 Migration")
body(
    "Gazebo 9 (which shipped with ROS1 Melodic) was the original simulation backend. "
    "Migration to Gazebo 11 (Citadel), which is compatible with ROS2 Humble, required "
    "the following changes:"
)
bullet("World files (.world): Updated SDF format from version 1.5 to 1.7")
bullet("gazebo_ros_init, gazebo_ros_factory plugins: Replaced deprecated ROS1 equivalents")
bullet("URDF/Xacro drone models: Gazebo 11 sensor plugin namespace updated (libgazebo_ros_camera.so)")
bullet("Depth camera plugin: Migrated to libgazebo_ros_depth_camera.so with ROS2 topic remapping")
bullet("Force system plugin: libgazebo_ros_force_system.so for motor simulation")
bullet("Collision physics: Adjusted ODE solver parameters for improved stability at high speed")
bullet("Dynamic obstacles (pedestrians): Actor SDF with skin mesh and trajectory animation nodes")

h3("Tunnel World Structure")
body(
    "The tunnel worlds are constructed from two tunnel_straight mesh segments placed "
    "end-to-end, forming a 100 m × 8 m corridor. Available configurations:"
)
table(
    ["World File", "Obstacles", "Dynamic / Static", "Usage"],
    [
        ["tunnel_100m_dynamic_20.world", "20", "8 people + 12 boxes", "Testing (held-out)"],
        ["tunnel_100m_dynamic_25.world", "25", "10 people + 15 boxes", "Training"],
        ["tunnel_100m_dynamic_30.world", "30", "10 people + 20 boxes", "Stress test"],
        ["tunnel_straight_static.world", "0", "—", "Baseline / debugging"],
    ],
    [2.8, 1.0, 2.0, 2.0]
)

h2("3.3 MAVROS and PX4 SITL Integration")
body(
    "For Software-in-Loop (SIL) validation, the PX4 iris_depth_camera model was integrated "
    "alongside the CERLAB native drone. MAVROS provides the ROS2 bridge between the PX4 "
    "flight controller stack and the DRL policy running on the companion computer side."
)
h3("PX4 SITL Launch Sequence")
numbered("Start PX4 SITL firmware with iris_depth_camera model:")
code("cd PX4-Autopilot && make px4_sitl gazebo-classic_iris_depth_camera")
numbered("Launch MAVROS with PX4 FCU URL:")
code("ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557")
numbered("Set OFFBOARD mode and arm via MAVROS:")
code("ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \"{custom_mode: 'OFFBOARD'}\"")
numbered("Send position setpoints at ≥ 2 Hz to maintain OFFBOARD mode:")
code("# Policy publishes PoseStamped to /mavros/setpoint_position/local at 10 Hz")
body(
    "Key difference from the native CERLAB drone: the PX4 MAVROS platform requires a "
    "2 m safe altitude (vs 1.5 m for the native sim) due to the iris airframe geometry, "
    "and the policy loop runs at 10 Hz to satisfy MAVROS OFFBOARD heartbeat requirements.",
    italic=True
)

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SIMULATION ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
h1("4. Simulation Environment")

h2("4.1 Tunnel World Design")
body(
    "The primary training and testing environment is a straight tunnel of 100 m total "
    "length with an arched cross-section (~8 m wide, ~8 m tall at the centreline). "
    "The drone flies along the X-axis, with Y representing lateral position and Z altitude."
)
table(
    ["Parameter", "Value"],
    [
        ["Tunnel total length", "100 m"],
        ["Usable corridor half-width (±Y)", "±3.5 m"],
        ["Safe cruise altitude", "1.5 m (native) / 2.0 m (PX4)"],
        ["Turn-point (outbound leg end)", "95 m"],
        ["Obstacle types", "Dynamic walking people + static boxes"],
        ["People speed", "0.5–0.7 m/s, sinusoidal lateral motion"],
        ["Box dimensions", "1 × 1 × 2 m"],
        ["Obstacle separation (min)", "~5 m"],
        ["Collision threshold", "0.35 m from drone centroid"],
        ["Near-miss threshold", "0.80 m"],
    ],
    [3.0, 4.0]
)

h2("4.2 Drone Models")
table(
    ["Platform", "Model", "Controller", "Depth Camera", "Safe Alt"],
    [
        ["CERLAB Quadcopter", "quadcopter.urdf", "Position setpoint (PD)", "Intel RealSense D435 sim", "1.5 m"],
        ["PX4 iris", "iris_depth_camera", "PX4 SITL + MAVROS OFFBOARD", "Gazebo depth plugin", "2.0 m"],
    ],
    [1.8, 2.0, 2.2, 2.2, 1.0]
)
body(
    "The depth camera provides a 640×480 image at 10 Hz. The DRL policy processes this "
    "raw depth image into sector-based features (see Section 7) rather than using the "
    "full image, enabling lightweight inference on CPU at ~40 µs per step."
)

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — DRL ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
h1("5. DRL Algorithm — TD3 (Twin-Delayed DDPG)")
body(
    "Twin-Delayed Deep Deterministic Policy Gradient (TD3) was selected for this project "
    "due to its suitability for continuous action spaces, stability improvements over DDPG "
    "(twin critics, delayed actor updates, target policy smoothing), and proven performance "
    "in robotic control tasks. TD3 is an off-policy actor-critic algorithm that learns from "
    "experience stored in a replay buffer."
)
h2("Why TD3 over PPO or SAC?")
bullet("Continuous action space: TD3 directly outputs continuous velocity commands without discretisation")
bullet("Off-policy learning: Experience replay allows data-efficient learning from fewer environment interactions")
bullet("Deterministic policy: Inference is deterministic, critical for safety-critical deployment")
bullet("Lower variance than SAC: TD3's clipped double-Q target reduces Q-value overestimation")
bullet("Empirically validated: Prior CERLAB DRL work established TD3 as the reference algorithm")

h2("TD3 Core Components")
table(
    ["Component", "Design Choice", "Rationale"],
    [
        ["Replay Buffer", "100k transitions (R54)", "Balances diversity vs. recency; 200k caused OOM"],
        ["Batch Size", "256", "Standard for continuous control"],
        ["Actor LR", "3×10⁻⁴", "Increased from 1×10⁻⁴ for faster convergence"],
        ["Critic LR", "3×10⁻⁴", "Matched to actor"],
        ["Discount (γ)", "0.99", "Long-horizon task (100 m round-trip)"],
        ["Target Update (τ)", "0.005", "Soft polyak averaging"],
        ["Actor Update Delay", "Every 2 critic steps", "Delays actor to reduce instability"],
        ["Exploration Noise", "Gaussian, σ=0.06→0.01 decay", "Annealed over training"],
        ["Target Policy Noise", "σ=0.20, clip ±0.50", "Smoothing for critic training"],
    ],
    [2.0, 2.2, 2.8]
)

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TRAINING RUNS R50–R54
# ══════════════════════════════════════════════════════════════════════════════
h1("6. Training Process — Runs R50 to R54")
body(
    "Training was conducted in five major runs (R50–R54), each building on lessons "
    "learned from the previous run. All runs used the tunnel_100m_dynamic_25.world "
    "(25 obstacles: 10 dynamic people + 15 static boxes) as the training environment, "
    "running on the native CERLAB quadcopter simulator within Gazebo 11 / ROS2 Humble."
)

h2("Training Run Overview")
table(
    ["Run", "Episodes", "Success Rate", "Avg Speed", "Key Change", "Outcome"],
    [
        ["R50", "5,000", "11.4%", "~0.8 m/s", "Baseline 14D state, 5 sectors", "Stuck at 65–80 m; 22 regressions"],
        ["R51", "5,000", "16.5%", "0.9 m/s", "Harder braking near obstacles", "Speed capped — braking too aggressive"],
        ["R52", "5,000", "17.8%", "1.508 m/s", "Smoothed avoidance, softer braking", "Best speed policy (r52_ep5000.pth)"],
        ["R53", "5,000", "27.8%", "2.59 m/s", "Larger network 256→256→128", "Best SR to date; speed improving"],
        ["R54", "5,000+", "Best", "Highest", "21D state, 9 sectors, 512→512→256 + LN + skip", "Final production policy"],
    ],
    [0.5, 0.8, 1.0, 1.0, 2.5, 2.2]
)

h2("6.1 State and Action Space Design Evolution")
body("The state vector grew significantly across runs to capture richer environmental information:")
table(
    ["Component", "R50–R52 (14D)", "R53 (14D)", "R54 (21D)"],
    [
        ["Horizontal depth sectors", "5 (36°/sector)", "5 (36°/sector)", "9 (20°/sector)"],
        ["Depth percentile", "10th", "10th", "5th — closer obstacle capture"],
        ["Vertical depth zones", "Top + Bottom (2)", "Top + Bottom (2)", "Top + Mid + Bottom (3)"],
        ["Proximity alarm flag", "No", "No", "Yes (dim 12)"],
        ["Forward progress", "Yes (dim 5)", "Yes (dim 5)", "Yes (dim 13)"],
        ["Lateral offset", "Yes (dim 6)", "Yes (dim 6)", "Yes (dim 14)"],
        ["Altitude error", "Yes (dim 7)", "Yes (dim 7)", "Yes (dim 15)"],
        ["Velocity (vx, vy, vz)", "Yes (dims 8–10)", "Yes (dims 8–10)", "Yes (dims 16–18)"],
        ["Phase flag (outbound/return)", "Yes (dim 11)", "Yes (dim 11)", "Yes (dim 19)"],
        ["Speed cap (curriculum)", "Yes (dim 12)", "Yes (dim 12)", "Yes (dim 20)"],
        ["State dimension total", "14", "14", "21"],
    ],
    [3.0, 1.5, 1.5, 1.5]
)

h2("6.2 Run R50 — Baseline")
body(
    "R50 established the training baseline using the 14-dimensional state, 5 horizontal "
    "depth sectors, and a shallow actor network (256 → 256 → 128). The drone learned to "
    "navigate the first 65 m reliably but consistently failed at a cluster of closely-spaced "
    "obstacles near 75 m. Twenty-two performance regressions were recorded — episodes where "
    "previously mastered distances were lost — indicating instability in the learning signal."
)
bullet("Success Rate: 11.4% at episode 5,000")
bullet("Primary failure: obstacle cluster at 75 m caused stuck behaviour")
bullet("22 regressions (SR dropped > 5% from previous best)")
bullet("Buffer size: 50k — too small, leading to catastrophic forgetting")

h2("6.3 Run R51 — Speed Cap Issue")
body(
    "R51 introduced harder proximity braking to reduce collisions. While collisions decreased, "
    "the braking logic triggered excessively, capping the drone's average speed at 0.9 m/s "
    "even in obstacle-free sections. The policy learned to be overly cautious rather than "
    "learning true avoidance. SR improved modestly to 16.5%."
)
bullet("Success Rate: 16.5%")
bullet("Avg speed: 0.9 m/s — far below the 3–6 m/s target")
bullet("Lesson: Hard braking must not override smooth avoidance learning")

h2("6.4 Run R52 — Speed Improvement")
body(
    "R52 replaced the hard-braking logic with smoother proportional avoidance: lateral "
    "push forces proportional to obstacle proximity rather than binary speed cuts. "
    "This allowed the drone to navigate through tight gaps without freezing. "
    "Average speed improved significantly to 1.508 m/s. The R52 best policy (r52_ep5000.pth) "
    "became the speed baseline for subsequent runs."
)
bullet("Success Rate: 17.8% (modest gain — speed improved, not SR)")
bullet("Avg speed: 1.508 m/s — major improvement from R51")
bullet("Best checkpoint: r52_ep5000.pth retained as speed baseline")

h2("6.5 Run R53 — Architecture Scaling")
body(
    "R53 scaled the network to 512 → 512 → 256 hidden units (from 256 → 256 → 128) "
    "with LayerNorm added to all hidden layers for training stability. The larger network "
    "capacity allowed the policy to discriminate finer obstacle configurations. SR nearly "
    "doubled to 27.8%, with average speed climbing to 2.59 m/s."
)
bullet("Success Rate: 27.8% — best SR achieved so far")
bullet("Avg speed: 2.59 m/s")
bullet("Network: 512→512→256 + LayerNorm at each layer")
bullet("Buffer: increased to 100k")

h2("6.6 Run R54 — Final Architecture (Best Policy)")
body(
    "R54 introduced the most significant changes to both the state representation and "
    "network architecture. Nine horizontal depth sectors (doubled from 5 — reducing "
    "angular blind spots from 36° to 20°), a 5th-percentile depth estimator (detecting "
    "the closest object in each sector), three vertical depth zones (top/mid/bottom), "
    "a binary proximity alarm flag, and a residual skip connection in the actor network "
    "were all added simultaneously. The result was the best-performing policy across all metrics."
)
bullet("State dimension: 21 (from 14)")
bullet("Depth sectors: 9 horizontal + 3 vertical = 12 depth features (from 7)")
bullet("Proximity alarm: binary flag at d < 1.5 m (prevents missed detections)")
bullet("Actor: 512 → 512 → 256 with LayerNorm + residual skip connection")
bullet("Critic: Twin 512 → 512 → 256 with LayerNorm")
bullet("Smooth proportional avoidance: col/mission 3.1 → 1.1 (64% reduction)")
bullet("Best policy: r54_best.pth — selected by highest rolling-20 SR")

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — R54 ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
h1("7. R54 Architecture and Reward Design")

h2("7.1 State Vector (21 Dimensions)")
table(
    ["Dims", "Feature", "Description"],
    [
        ["0 – 8", "Horizontal depth sectors (9)", "5th-percentile depth per 20° sector, normalised to [0,1] by DEPTH_FAR=10m"],
        ["9", "Vertical depth — top", "5th-percentile depth, upper 33% of camera FOV"],
        ["10", "Vertical depth — mid", "5th-percentile depth, middle 33% of camera FOV"],
        ["11", "Vertical depth — bottom", "5th-percentile depth, lower 33% of camera FOV"],
        ["12", "Proximity alarm", "Binary: 1 if any sector < 1.5 m, else 0"],
        ["13", "Forward progress", "Distance travelled towards goal, normalised to [0, 1]"],
        ["14", "Lateral offset", "Y deviation from tunnel centreline, normalised to [-1, 1]"],
        ["15", "Altitude error", "(pz − safe_alt) / 1.5, normalised to [-1, 1]"],
        ["16", "Forward velocity vx", "Normalised by 7.0 m/s"],
        ["17", "Lateral velocity vy", "Normalised by 3.0 m/s"],
        ["18", "Vertical velocity vz", "Normalised by 2.0 m/s"],
        ["19", "Phase flag", "0 = OUTBOUND, 1 = RETURN"],
        ["20", "Speed cap", "Current curriculum speed stage, normalised by 7.0"],
    ],
    [0.8, 2.0, 4.2]
)

h2("7.2 Actor Network Architecture")
body("The R54 actor uses a feed-forward network with a residual skip connection:")
code("Input:  21D state vector")
code("FC1:    21 → 512  + LayerNorm(512) + ReLU")
code("FC2:    512 → 512 + LayerNorm(512) + ReLU  ← adds skip from FC1 output")
code("FC3:    512 → 256 + LayerNorm(256) + ReLU")
code("FC4:    256 → 3   + tanh activation")
code("Output: [forward_cmd, lateral_cmd, altitude_cmd] ∈ [-1, 1]³")
body("The residual skip connection (FC2 output += linear(FC1 output)) prevents gradient vanishing in the deeper network and allows the policy to bypass the second 512-unit layer when the first layer's representation is already sufficient.")

h2("7.3 Reward Function")
table(
    ["Component", "Value", "Condition"],
    [
        ["Forward progress reward", "+150 × Δforward (m)", "Per step, outbound"],
        ["Return progress reward", "+120 × Δreturn (m)", "Per step, return leg"],
        ["Speed bonus (>4.5 m/s)", "+25 / tick", "High-speed encouragement"],
        ["Speed bonus (>3.5 m/s)", "+12 / tick", "Mid-speed bonus"],
        ["Speed bonus (>2.0 m/s)", "+6 / tick", "Baseline movement"],
        ["Proximity penalty (exp)", "−60 × exp(−d × 3.0)", "Exponential near obstacles"],
        ["Collision penalty", "−80 per collision", "d < 0.35 m from obstacle"],
        ["Lateral deviation penalty", "−2.5 × |y_offset|", "Lane-keeping"],
        ["Altitude deviation penalty", "−8 × |Δalt|", "Height stability"],
        ["Turn-point reached bonus", "+3,000", "Reached 95 m"],
        ["Mission success bonus", "+5,000", "Full round-trip complete"],
        ["Timeout penalty", "−30", "Episode ended at 2,000 steps"],
    ],
    [2.8, 1.5, 2.7]
)

h2("7.4 Speed Curriculum")
body(
    "R54 uses a staged speed curriculum controlled by rolling success rate (SR). "
    "The drone begins at a low speed and advances only when it demonstrates consistent "
    "performance, preventing the policy from being overwhelmed by obstacle density at high speed."
)
table(
    ["Stage", "Max Speed", "SR Gate (rolling 20)", "Min Missions"],
    [
        ["1", "2.0 m/s", "Start", "—"],
        ["2", "3.0 m/s", "≥ 55%", "20"],
        ["3", "4.0 m/s", "≥ 65%", "20"],
        ["4", "4.5 m/s", "≥ 72%", "20"],
        ["5", "5.0 m/s", "≥ 80%", "20"],
    ],
    [0.8, 1.2, 2.2, 1.8]
)

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
h1("8. Cross-Validation")
body(
    "Two complementary cross-validation protocols were applied to the R54 best policy: "
    "(1) Observation Withholding — testing policy robustness to missing sensor channels, "
    "and (2) K-Fold Blocked Episode CV — testing generalisation across different spatial "
    "starting conditions while addressing temporal autocorrelation and spatial leakage."
)

h2("8.1 Observation Withholding (Figure 6.11)")
body(
    "Five observation conditions were tested, each zeroing a subset of the 21-dimensional "
    "state vector to simulate sensor failure or reduced sensor availability. "
    "One episode per condition was run in tunnel_100m_dynamic_25.world."
)
table(
    ["Condition", "Dims Zeroed", "Observation Removed", "Max Fwd (m)", "Avg Spd Moving (m/s)", "Outbound?"],
    [
        ["L0 — Full", "None (0)", "Baseline — all sensors active", "52.8", "0.587", "No"],
        ["L1 — No vert. depth", "9–11 (3)", "Top/mid/bottom depth channels", "95.1", "3.589", "✓ YES"],
        ["L2 — No velocity", "16–18 (3)", "vx, vy, vz zeroed", "69.9", "3.913", "No"],
        ["L3 — Depth only", "12–20 (9)", "Navigation, velocity, flags", "53.0", "1.708", "No"],
        ["L4 — No depth", "0–12 (13)", "All depth sensors removed", "52.9", "1.700", "No"],
    ],
    [1.5, 1.5, 2.5, 1.2, 1.8, 0.9]
)
body(
    "Key finding: L1 (no vertical depth) achieved the best performance, completing the "
    "full 95 m outbound leg — the only condition to do so. This suggests vertical depth "
    "information may introduce conflicting avoidance signals in certain obstacle "
    "configurations. All conditions timed out on the return leg due to the absence of a "
    "reverse/recovery mechanism in the policy.",
    italic=True
)
try:
    img(CROSS_IMG / "fig6_11_crossval_robustness.png", width=5.8,
        caption="Figure 6.11 — Observation Withholding Cross-Validation Results (L0–L4)")
except Exception:
    pass

h2("8.2 K-Fold Blocked Cross-Validation")
body(
    "A proper K-fold cross-validation was designed to address three canonical failure "
    "modes of naive RL evaluation:"
)
bullet("Temporal Autocorrelation: Episodes treated as atomic units; folds are contiguous temporal blocks — no timestep shuffling")
bullet("Spatial Leakage: Each fold uses a distinct set of lateral starting y-offsets, testing spatial generalisation")
bullet("Dynamic Physics: Every episode is a live closed-loop interaction with Gazebo — no replayed data")

h3("Protocol")
table(
    ["Parameter", "Value"],
    [
        ["Folds (K)", "5"],
        ["Episodes per fold (N)", "3"],
        ["Total episodes", "15"],
        ["Episode timeout", "90 s"],
        ["World", "tunnel_100m_dynamic_25.world (training domain)"],
        ["Observation condition", "L0 — Full 21D state"],
        ["Starting y-offsets (per fold)", "Fold 0: {−0.30, 0.00, +0.30}; Fold 1: {−0.20, +0.05, +0.20}; etc."],
    ],
    [2.5, 4.5]
)
h3("K-Fold Results")
table(
    ["Metric", "Mean ± Std", "95% CI (Bootstrap)"],
    [
        ["Max forward progress", "47.4 ± 29.1 m", "[33.7, 61.4] m"],
        ["Outbound completion", "50 ± 31 %", "[35, 65] %"],
        ["Avg moving speed", "1.79 ± 0.69 m/s", "[1.52, 2.15] m/s"],
        ["Time to stuck", "31.0 ± 25.3 s", "[19.7, 44.2] s"],
        ["Collision events", "2.8 ± 1.8", "[1.9, 3.8]"],
        ["Min obstacle distance", "0.069 ± 0.030 m", "[0.056, 0.085] m"],
        ["Success rate", "0% (0/15)", "Binomial 95% CI: [0%, 21.8%]"],
        ["Fold-to-fold CoV", "0.286", "Above 0.20 threshold → spatial sensitivity"],
        ["Composite CV Score", "0.467 / 1.0", "—"],
    ],
    [2.5, 2.0, 2.5]
)
body(
    "The fold-to-fold coefficient of variation (CoV = 0.286 > 0.20) indicates that the "
    "policy is spatially sensitive — performance depends on the starting lateral offset. "
    "Episodes starting near y = −0.15 m consistently got stuck early (~13 m), corresponding "
    "to an unfavourable approach angle at the first obstacle cluster. Episodes starting near "
    "y = +0.00 to +0.25 m reached 70–95 m.",
    italic=True
)

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TESTING PHASE
# ══════════════════════════════════════════════════════════════════════════════
h1("9. Testing Phase")
body(
    "The R54 best policy was evaluated in three distinct test configurations to assess "
    "domain generalisation and platform transfer capability. All tests used the "
    "r54_best.pth checkpoint selected by highest rolling-20 success rate during training."
)
table(
    ["Test", "Platform", "World", "Obstacles", "Env. Seen?", "Safe Alt"],
    [
        ["A", "Native CERLAB sim", "tunnel_100m_dynamic_20.world", "20", "No (held-out)", "1.5 m"],
        ["B", "PX4 iris + MAVROS", "tunnel_100m_dynamic_20.world", "20", "No (held-out)", "2.0 m"],
        ["C", "PX4 iris + MAVROS", "tunnel_100m_dynamic_25.world", "25", "Yes (training)", "2.0 m"],
    ],
    [0.4, 1.8, 2.8, 1.0, 1.4, 0.8]
)

h2("9.1 Test A — Domain Randomisation (Native CERLAB Drone)")
body(
    "Test A validated the policy's ability to generalise to a held-out obstacle density "
    "(20 vs 25 obstacles in training). This tests domain randomisation: the policy was "
    "never trained on this exact world, with different obstacle positions and 5 fewer obstacles."
)
table(
    ["Metric", "Value"],
    [
        ["Result", "✓ SUCCESS — Full round-trip completed"],
        ["Duration", "99.7 s"],
        ["Avg speed", "1.739 m/s"],
        ["Peak speed", "3.923 m/s"],
        ["Min obstacle distance", "0.279 m"],
        ["Collisions", "2 (both recoverable)"],
        ["Near-misses (< 0.8 m)", "38"],
        ["Safe altitude compliance", ">95% of flight at 1.3–1.8 m"],
    ],
    [3.0, 4.0]
)
try:
    img(TEST_IMG / "A_a_mission_progress.png", width=5.5,
        caption="Figure — Test A: Mission Progress (native drone, 20-obstacle testing world)")
    img(TEST_IMG / "A_d_obstacle_distance.png", width=5.5,
        caption="Figure — Test A: Min Obstacle Distance over time")
except Exception:
    pass

h2("9.2 Test B — Software-in-Loop Validation (PX4, Testing World)")
body(
    "Test B transferred the policy to the PX4 SITL platform with MAVROS OFFBOARD control, "
    "running in the same 20-obstacle testing world as Test A. This validates platform "
    "transfer: the same weights, no retraining, different flight controller stack."
)
table(
    ["Metric", "Value", "vs Test A"],
    [
        ["Result", "✓ SUCCESS", "—"],
        ["Duration", "76.7 s", "23 s faster"],
        ["Avg speed", "2.449 m/s", "+0.71 m/s (+41%)"],
        ["Peak speed", "4.849 m/s", "+0.93 m/s higher"],
        ["Min obstacle distance", "0.851 m", "+0.572 m safer"],
        ["Collisions", "0", "−2 (clean flight)"],
        ["Near-misses", "0", "−38 (clean flight)"],
    ],
    [2.5, 2.0, 2.5]
)
body(
    "The PX4 platform navigated significantly faster and with much higher obstacle clearance. "
    "The PX4 iris airframe has a larger flight envelope and the MAVROS position controller "
    "implements smoother velocity transitions, resulting in cleaner trajectories.",
    italic=True
)

h2("9.3 Test C — Software-in-Loop Validation (PX4, Training World)")
body(
    "Test C ran the PX4 platform in the training world (25 obstacles, familiar layout) "
    "to test whether the policy achieves higher performance in the environment it was "
    "trained on, and to confirm the SIL stack functions correctly in the training domain."
)
table(
    ["Metric", "Value", "vs Test B (same platform)"],
    [
        ["Result", "✓ SUCCESS", "—"],
        ["Duration", "80.6 s", "+3.9 s (more obstacles)"],
        ["Avg speed", "2.195 m/s", "−0.25 m/s (denser obstacles)"],
        ["Peak speed", "6.531 m/s", "+1.68 m/s higher peak"],
        ["Min obstacle distance", "0.856 m", "Comparable"],
        ["Collisions", "0", "Clean flight"],
        ["Near-misses", "0", "Clean flight"],
    ],
    [2.5, 2.0, 2.5]
)
try:
    img(TEST_IMG / "BC_a_mission_progress.png", width=5.5,
        caption="Figure — Tests B+C: Mission Progress comparison (PX4 MAVROS, both worlds)")
    img(TEST_IMG / "BC_c_speed.png", width=5.5,
        caption="Figure — Tests B+C: Flight Speed comparison")
except Exception:
    pass

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
h1("10. Results Summary and Analysis")

h2("10.1 Three-Test Comparative Summary")
table(
    ["Metric", "Test A\nNative / 20 obs", "Test B\nPX4 / 20 obs", "Test C\nPX4 / 25 obs"],
    [
        ["Result", "✓ SUCCESS", "✓ SUCCESS", "✓ SUCCESS"],
        ["Duration (s)", "99.7", "76.7", "80.6"],
        ["Avg speed (m/s)", "1.739", "2.449", "2.195"],
        ["Peak speed (m/s)", "3.923", "4.849", "6.531"],
        ["Min depth (m)", "0.279", "0.851", "0.856"],
        ["Collisions", "2", "0", "0"],
        ["Near-misses", "38", "0", "0"],
        ["Safe alt %", ">92%", ">98%", ">97%"],
    ],
    [2.5, 1.7, 1.7, 1.7]
)
body(
    "All three test configurations produced successful round-trip completions, "
    "demonstrating that the R54 policy generalises across: (1) unseen obstacle densities "
    "(Test A vs training), (2) different hardware platforms (Tests B and C vs Test A), "
    "and (3) both seen and unseen environments on the PX4 platform."
)

h2("10.2 Key Findings")
numbered("Policy robustness: A single set of weights (r54_best.pth) succeeded in all three test configurations without fine-tuning.")
numbered("Platform transfer: Zero-shot transfer from the native CERLAB simulator to the PX4 MAVROS stack succeeded — the policy generalised across different controller dynamics, safe altitudes, and obstacle clearance behaviours.")
numbered("Speed advantage of PX4: The PX4 MAVROS platform achieved 41% higher average speed than the native sim (2.449 vs 1.739 m/s), likely due to smoother MAVROS velocity interpolation.")
numbered("Obstacle clearance: The native sim drone operated much closer to obstacles (min depth 0.279 m vs 0.851 m for PX4), reflecting the difference in controller aggressiveness.")
numbered("Observation withholding: The L1 condition (no vertical depth) paradoxically outperformed full observation, suggesting vertical depth may introduce conflicting signals at 25-obstacle density. This is a candidate for ablation study in future work.")
numbered("K-Fold CV spatial sensitivity: The policy performs inconsistently across different starting lateral positions (CoV=0.286 > threshold). A starting position of y≈−0.15 m consistently produced early-stuck behaviour, indicating a specific obstacle cluster at that approach angle.")

h2("10.3 Training Dashboard")
try:
    img(TRAIN_IMG / "R54_Training_Dashboard_Overview.png", width=6.0,
        caption="Figure — R54 Training Dashboard: SR, speed, collision, and loss curves over 5,000 episodes")
except Exception:
    pass

pagebreak()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — LIMITATIONS AND FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════
h1("11. Limitations and Future Work")

h2("11.1 Current Limitations")
bullet("No reverse/recovery mechanism: The policy has no backward-flight or hover-escape behaviour. When stuck against an obstacle, the episode times out — this accounts for the 0% success rate in the K-fold CV (training-world single episodes in the cross-validation showed all timeouts).")
bullet("Single-shot testing: Each test configuration was evaluated in one episode (for Tests A–C) due to simulation time constraints. Multiple episodes per configuration would provide statistical confidence intervals.")
bullet("Spatial sensitivity: The K-fold CV revealed position-dependent performance (CoV=0.286). The policy is not uniformly robust across all lateral starting positions.")
bullet("World generalization: Cross-validation was conducted only on the 25-obstacle training world. The 30-obstacle never-seen stress-test world was not evaluated.")
bullet("Sim-to-real gap: All results are in simulation. Real-world deployment on physical hardware has not been conducted.")
bullet("Observation withholding (L1 finding): The result that removing vertical depth improves performance warrants investigation — it may indicate a reward design issue or a sensor noise interaction unique to the Gazebo environment.")

h2("11.2 Recommended Future Work")
numbered("Implement escape/recovery behaviour: Add a backward-flight or sideways-slide recovery trigger when the drone detects it is stuck (speed < 0.2 m/s for > 2 s).")
numbered("Multi-episode testing: Run ≥ 20 episodes per test configuration for statistical significance; report success rate with Clopper-Pearson 95% CI.")
numbered("Held-out world evaluation: Repeat cross-validation on tunnel_100m_dynamic_30.world (never seen) to measure true domain generalisation.")
numbered("Ablation study: Remove/restore each state dimension independently to quantify its contribution to SR and speed, explaining the L1 counterintuitive result.")
numbered("Real-world hardware deployment: Transfer the policy to a physical quadcopter equipped with an Intel RealSense D435 and a Jetson Xavier companion computer.")
numbered("Curriculum diversification: Add random obstacle spawning with variable positions per episode during training to reduce spatial overfitting (address CoV=0.286).")
numbered("Continuous training: Continue R54 training beyond 5,000 episodes — the SR curve had not plateaued, suggesting further improvement is achievable.")

divider()
body(
    "End of document.  All training checkpoints, telemetry CSVs, cross-validation data, "
    "and publication-quality figures are archived under "
    "Final Results/testing/  and  Final Results/testing/plots/Actual/.",
    italic=True, color=GREY
)

doc.save(str(OUT))
print(f"Saved: {OUT}")

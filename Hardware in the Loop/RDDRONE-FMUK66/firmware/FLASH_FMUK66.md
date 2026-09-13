# Flashing PX4 HITL Firmware onto RDDRONE-FMUK66

## Method A — QGroundControl (easiest, recommended)

1. Download and open QGroundControl:
   ```bash
   ~/QGroundControl.AppImage
   ```
2. Connect FMUK66 via USB while QGC is open.
3. QGC → Vehicle Setup → Firmware tab.
4. Select **PX4 Pro Stable Release** → click **OK**.
5. QGC flashes the correct `nxp_fmuk66-v3_default` binary automatically.
6. Wait for the progress bar and the "Upgrade complete" message.

---

## Method B — Build from source and flash (advanced)

Use this if you need a custom PX4 build (e.g., enabling a specific module or
patching HITL behaviour).

### Prerequisites (installed by install.sh)
```
gcc-arm-none-eabi    ← cross-compiler for ARM Cortex-M4
cmake, ninja-build
python3 empy toml packaging kconfiglib
```

### Build steps
```bash
cd ~/PX4-Autopilot

# Make sure submodules are up to date
git submodule update --init --recursive

# Build for FMUK66-v3
make nxp_fmuk66-v3_default

# Build for FMUK66-e (if you have the newer 'e' variant)
# make nxp_fmuk66-e_default
```
Build output: `build/nxp_fmuk66-v3_default/nxp_fmuk66-v3_default.px4`

### Flash via USB (bootloader mode)
```bash
# Connect FMUK66 via USB, then:
make nxp_fmuk66-v3_default upload
```
The board must be in bootloader mode — hold the BOOT button while connecting
USB, or trigger via: `nsh> reboot -b` in the NuttX shell.

### Flash via QGroundControl (custom firmware)
1. Open QGC → Firmware → select "Advanced Settings" → "Custom firmware file"
2. Browse to: `~/PX4-Autopilot/build/nxp_fmuk66-v3_default/nxp_fmuk66-v3_default.px4`
3. Click OK.

---

## Verify firmware version

After flashing, open a serial terminal:
```bash
screen /dev/ttyACM0 57600
```
Type `ver all` to see the firmware version. You should see:
```
PX4 FMUK66-v3 ...
```
Exit screen: `Ctrl+A` then `K` then `Y`.

---

## FMUK66 variant identification

| Variant | Board marking | PX4 target |
|---------|--------------|------------|
| FMUK66-v3 | "RDDRONE-FMUK66" (white PCB) | `nxp_fmuk66-v3_default` |
| FMUK66-e  | "RDDRONE-FMUK66E" (newer)    | `nxp_fmuk66-e_default`  |

If unsure, use QGroundControl — it auto-detects the correct binary.

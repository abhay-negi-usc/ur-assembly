#!/usr/bin/env bash
# Build and flash the toolchanger firmware to an Arduino Uno, with ONLY the selected modules
# compiled in, using the toolchain that ships with the Arduino IDE (~/.arduino15).
#
#   ./build_flash.sh                          build the config's modules (safe, touches no hardware)
#   ./build_flash.sh upload                   build them, then flash the board
#   ./build_flash.sh --modules screwdrive     build just these (comma-separated), ignoring the config
#   ./build_flash.sh upload --config PATH     take the module list from another deployment's config
#   ./build_flash.sh upload --port /dev/ttyACM2   (or PORT=/dev/ttyACM2 ./build_flash.sh upload)
#
# The module list comes from `modules:` in ../config/multitoolchanger.yaml unless --modules is
# given -- the same list the script loads, so the board and the script agree. The board
# announces what it was built with, and the script loads exactly that unless run --no-detect.
#
# Each module's pins come from the `pins:` block of its config/<module>.yaml, and are checked
# against the board and each other before anything is compiled (mtc/pins.py).
set -euo pipefail

SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${MULTITOOLCHANGER_CONFIG:-$SRC/../config/multitoolchanger.yaml}
MODULES=
UPLOAD=
PORT=${PORT:-/dev/ttyACM1}

while [ $# -gt 0 ]; do
  case $1 in
    upload)     UPLOAD=1 ;;
    --modules)  MODULES=${2:?--modules needs a comma-separated list}; shift ;;
    --config)   CONFIG=${2:?--config needs a path}; shift ;;
    --port)     PORT=${2:?--port needs a device}; shift ;;
    -h|--help)  sed -n '2,16p' "$0"; exit 0 ;;
    *)          echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# Which modules, and on which pins. mtc/pins.py reads the module list (the config's, or
# --modules) and each module's `pins:` from its yaml, REFUSES pins that clash or cannot do what
# the module needs (PWM, an interrupt, a timer another module took over), and prints the module
# names and the -DPIN_<MODULE>_<ROLE> flags that compile the pins in.
CONFIG=$(realpath -m "$CONFIG")
ARGS=(--config "$CONFIG" --firmware "$SRC")
[ -n "$MODULES" ] && ARGS+=(--modules "$MODULES")
PLAN=$(cd "$SRC/.." && python3 -m mtc.pins "${ARGS[@]}") || exit 2
SELECTED=$(sed -n 1p <<< "$PLAN")
DEFINES=$(sed -n 2p <<< "$PLAN")
for m in $SELECTED; do
  DEFINES="$DEFINES -DMODULE_${m^^}"
done
if [ -n "$MODULES" ]; then FROM="from --modules"; else FROM="from $CONFIG"; fi

A=~/.arduino15/packages/arduino
AVRBIN=$A/tools/avr-gcc/7.3.0-atmel3.6.1-arduino7/bin
DUDE=$A/tools/avrdude/6.3.0-arduino17
CORE=$A/hardware/avr/1.8.6/cores/arduino
VAR=$A/hardware/avr/1.8.6/variants/standard
SERVO=~/.arduino15/libraries/Servo/src
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT

COMMON="-Os -ffunction-sections -fdata-sections -mmcu=atmega328p -DF_CPU=16000000L \
-DARDUINO=10819 -DARDUINO_AVR_UNO -DARDUINO_ARCH_AVR -I$CORE -I$VAR -I$SERVO"
CF="-std=gnu11 -flto -fno-fat-lto-objects -w"
CXF="-std=gnu++11 -fpermissive -fno-exceptions -fno-threadsafe-statics -flto"

echo "==> modules: ${SELECTED// /, } ($FROM)"

echo "==> compiling core"
for f in "$CORE"/*.c;   do $AVRBIN/avr-gcc -c $COMMON $CF "$f" -o "$B/c_$(basename "$f").o"; done
for f in "$CORE"/*.cpp; do $AVRBIN/avr-g++ -c $COMMON $CXF -w "$f" -o "$B/c_$(basename "$f").o"; done

# only the coupler drives a servo
case " $SELECTED " in *" coupler "*)
  echo "==> compiling Servo"
  for f in "$SERVO"/*.cpp "$SERVO"/avr/*.cpp; do
    [ -f "$f" ] && $AVRBIN/avr-g++ -c $COMMON $CXF -w "$f" -o "$B/s_$(basename "$f").o"
  done ;;
esac

echo "==> compiling main.cpp ${SELECTED// /.cpp }.cpp (warnings on)"
for f in main $SELECTED; do
  $AVRBIN/avr-g++ -c $COMMON $CXF $DEFINES -Wall -Wextra "$SRC/$f.cpp" -o "$B/m_$f.o"
done

echo "==> linking"
$AVRBIN/avr-gcc -Os -flto -fuse-linker-plugin -Wl,--gc-sections -mmcu=atmega328p \
  -o "$B/fw.elf" "$B"/m_*.o $(ls "$B"/s_*.o 2>/dev/null) "$B"/c_*.o -lm
$AVRBIN/avr-objcopy -O ihex -R .eeprom "$B/fw.elf" "$B/fw.hex"
$AVRBIN/avr-size --mcu=atmega328p -C "$B/fw.elf"

if [ -n "$UPLOAD" ]; then
  echo "==> flashing $PORT with ${SELECTED// /, } (a coupler's servo WILL move on reset)"
  "$DUDE/bin/avrdude" -C "$DUDE/etc/avrdude.conf" -patmega328p -carduino \
    -P"$PORT" -b115200 -D -Uflash:w:"$B/fw.hex":i
  echo "==> done"
else
  echo "==> build only. Run './build_flash.sh upload' to flash."
fi

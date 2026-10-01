#!/usr/bin/env bash
# Build and flash firmware/main.cpp (T74 + IBT-2 timed-angle sketch) to an Arduino Uno,
# using the toolchain that ships with the Arduino IDE (~/.arduino15). No PlatformIO, no IDE.
#
#   ./build_flash.sh          build only (safe, touches no hardware)
#   PORT=/dev/ttyACM0 ./build_flash.sh upload   build then flash (PORT is required:
#                                              several Unos share this PC, run ../t74.py list)
set -euo pipefail

A=~/.arduino15/packages/arduino
AVRBIN=$A/tools/avr-gcc/7.3.0-atmel3.6.1-arduino7/bin
DUDE=$A/tools/avrdude/6.3.0-arduino17
CORE=$A/hardware/avr/1.8.6/cores/arduino
VAR=$A/hardware/avr/1.8.6/variants/standard
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=${PORT:-}
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT

COMMON="-Os -ffunction-sections -fdata-sections -mmcu=atmega328p -DF_CPU=16000000L \
-DARDUINO=10819 -DARDUINO_AVR_UNO -DARDUINO_ARCH_AVR -I$CORE -I$VAR"
CF="-std=gnu11 -flto -fno-fat-lto-objects -w"
CXF="-std=gnu++11 -fpermissive -fno-exceptions -fno-threadsafe-statics -flto"

echo "==> compiling core"
for f in "$CORE"/*.c;   do $AVRBIN/avr-gcc -c $COMMON $CF "$f" -o "$B/c_$(basename "$f").o"; done
for f in "$CORE"/*.cpp; do $AVRBIN/avr-g++ -c $COMMON $CXF -w "$f" -o "$B/c_$(basename "$f").o"; done

echo "==> compiling $SRC/*.cpp (warnings on)"
for f in "$SRC"/*.cpp; do
  $AVRBIN/avr-g++ -c $COMMON $CXF -Wall -Wextra "$f" -o "$B/m_$(basename "$f").o"
done

echo "==> linking"
$AVRBIN/avr-gcc -Os -flto -fuse-linker-plugin -Wl,--gc-sections -mmcu=atmega328p \
  -o "$B/fw.elf" "$B"/m_*.o "$B"/c_*.o -lm
$AVRBIN/avr-objcopy -O ihex -R .eeprom "$B/fw.elf" "$B/fw.hex"
$AVRBIN/avr-size --mcu=atmega328p -C "$B/fw.elf"

if [ "${1:-}" = "upload" ]; then
  if [ -z "$PORT" ]; then
    echo "==> set PORT, e.g. PORT=/dev/ttyACM0 ./build_flash.sh upload (see ../t74.py list)" >&2
    exit 1
  fi
  echo "==> flashing $PORT (motor pins are held LOW through setup)"
  "$DUDE/bin/avrdude" -C "$DUDE/etc/avrdude.conf" -patmega328p -carduino \
    -P"$PORT" -b115200 -D -Uflash:w:"$B/fw.hex":i
  echo "==> done"
else
  echo "==> build only. Run 'PORT=/dev/ttyACMx ./build_flash.sh upload' to flash."
fi

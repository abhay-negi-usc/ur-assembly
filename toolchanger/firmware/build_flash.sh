#!/usr/bin/env bash
# Build and flash firmware/main.cpp to an Arduino Uno, using the toolchain that
# ships with the Arduino IDE (~/.arduino15). No PlatformIO, no Arduino IDE needed.
#
#   ./build_flash.sh          build only (safe, touches no hardware)
#   ./build_flash.sh upload   build then flash the board
set -euo pipefail

A=~/.arduino15/packages/arduino
AVRBIN=$A/tools/avr-gcc/7.3.0-atmel3.6.1-arduino7/bin
DUDE=$A/tools/avrdude/6.3.0-arduino17
CORE=$A/hardware/avr/1.8.6/cores/arduino
VAR=$A/hardware/avr/1.8.6/variants/standard
SERVO=~/.arduino15/libraries/Servo/src
SKETCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/main.cpp
PORT=${PORT:-/dev/ttyACM0}
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT

COMMON="-Os -ffunction-sections -fdata-sections -mmcu=atmega328p -DF_CPU=16000000L \
-DARDUINO=10819 -DARDUINO_AVR_UNO -DARDUINO_ARCH_AVR -I$CORE -I$VAR -I$SERVO"
CF="-std=gnu11 -flto -fno-fat-lto-objects -w"
CXF="-std=gnu++11 -fpermissive -fno-exceptions -fno-threadsafe-statics -flto"

echo "==> compiling core + Servo"
for f in "$CORE"/*.c;   do $AVRBIN/avr-gcc -c $COMMON $CF "$f" -o "$B/c_$(basename "$f").o"; done
for f in "$CORE"/*.cpp; do $AVRBIN/avr-g++ -c $COMMON $CXF -w "$f" -o "$B/c_$(basename "$f").o"; done
for f in "$SERVO"/*.cpp "$SERVO"/avr/*.cpp; do
  [ -f "$f" ] && $AVRBIN/avr-g++ -c $COMMON $CXF -w "$f" -o "$B/s_$(basename "$f").o"
done

echo "==> compiling $SKETCH (warnings on)"
$AVRBIN/avr-g++ -c $COMMON $CXF -Wall -Wextra "$SKETCH" -o "$B/main.o"

echo "==> linking"
$AVRBIN/avr-gcc -Os -flto -fuse-linker-plugin -Wl,--gc-sections -mmcu=atmega328p \
  -o "$B/fw.elf" "$B/main.o" "$B"/s_*.o "$B"/c_*.o -lm
$AVRBIN/avr-objcopy -O ihex -R .eeprom "$B/fw.elf" "$B/fw.hex"
$AVRBIN/avr-size --mcu=atmega328p -C "$B/fw.elf"

if [ "${1:-}" = "upload" ]; then
  echo "==> flashing $PORT (the servo WILL move on reset)"
  "$DUDE/bin/avrdude" -C "$DUDE/etc/avrdude.conf" -patmega328p -carduino \
    -P"$PORT" -b115200 -D -Uflash:w:"$B/fw.hex":i
  echo "==> done"
else
  echo "==> build only. Run './build_flash.sh upload' to flash."
fi

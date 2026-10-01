#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
// T74 + IBT-2 + Elegoo Uno R3, with an optional AMT10E2-V quadrature encoder on the drive shaft.
// IBT-2: RPWM D5, LPWM D6, R_EN D7, L_EN D8.
// AMT10E2-V: A D2 (INT0), B D3 (INT1), X/index D4 (pin-change interrupt), 5V, GND.
// C = start one-full-tile-turn calibration at a physical mark.
// M = stop at the mark and save the measured turn time AND encoder counts (until reset/power-off).
// Then type a relative angle such as 60, 15.5, or -30. S = stop.
//   With an encoder (calibration saw counts): closed loop, the motor stops on the encoder count.
//   Without one (no counts during calibration): timed estimate, as before.
// T<ms> = load a turn time measured earlier (e.g. T4200), so a reset does not force a recal.
// K<counts> = load encoder counts per tile turn measured earlier (signed, e.g. K20480).
// P<pwm> = set the run speed, 1..255 (e.g. P80). A new speed clears the turn time (not the counts).
// H = home: run forward to the encoder index pulse and call that 0 deg.
// G<deg> = go to an absolute angle from home (or from where the board started, if not homed).
// E = print the encoder count and angle.
// ? = print the current turn time, counts, run speed and mode.
// Opening the serial port resets the Uno, which clears the turn time, counts, speed and home.
// t74.py saves the first three to a file and sends them back after every reset.
const byte RPWM = 5, LPWM = 6, REN = 7, LEN = 8;
const byte ENC_A = 2, ENC_B = 3, ENC_X = 4;  // A/B must be D2/D3: the Uno's only INT0/INT1 pins.
const unsigned int ENC_PPR = 5120;           // AMT10E DIP switch (factory, all off = 5120 PPR).
const byte DEFAULT_PWM = 45;               // Run speed after a reset, until a P<pwm> arrives.
const unsigned long MIN_CAL_MS = 500UL;
const unsigned long MAX_CAL_MS = 60000UL; // Stops unattended calibration.
const unsigned long MAX_MOVE_MS = 60000UL;
const long MIN_TURN_COUNTS = 100;          // Fewer counts in a calibration turn = no encoder.
const long MAX_TURN_COUNTS = 10000000L;
const unsigned long STALL_MS = 1000UL;     // Motor on, encoder silent this long = fault.
const unsigned long SETTLE_MS = 150UL;     // Coasting is over once no counts arrive this long.
const unsigned long MAX_SETTLE_MS = 2000UL;
const unsigned long VELOCITY_MS = 5UL;     // Speed is measured over this window.
const byte MAX_CORRECTIONS = 3;            // Short moves back toward the target after an overshoot.
enum Mode { IDLE, CALIBRATING, MOVING, ENC_MOVING, HOMING, SETTLING };
Mode mode = IDLE;
byte runPwm = DEFAULT_PWM;                 // Same PWM used for calibration and moves.
unsigned long started = 0, fullTurnMs = 0, moveMs = 0;
char input[24];
byte inputLength = 0;
bool inputOverflow = false;
char pending = 0;                          // 'T', 'P', 'K' or 'G' if the number follows one.
unsigned long lastInputByte = 0;

// ---- encoder state ----
volatile long encCount = 0;                // Quadrature counts (4 per PPR), signed.
volatile long indexCount = 0;              // encCount at the last index rising edge.
volatile unsigned int indexPulses = 0;
volatile unsigned int missedEdges = 0;     // Both channels changed at once: too fast or noise.
volatile byte encState = 0;
long turnCounts = 0;          // Signed counts for one FORWARD tile turn. 0 = no encoder: timer only.
long zeroCount = 0;           // Count that means 0 deg: the index after H, else the boot position.
bool homed = false;
long setpoint = 0;            // Absolute target count; relative moves add to it so errors don't add up.
bool setpointValid = false;   // False after anything that moved the tile without a target.
long calStartCount = 0, moveStartCount = 0, moveMagnitude = 0, stopProgress = 0;
int8_t moveDir = 1;           // +1/-1: which way the count must go for this move.
// Coast after cutting power is about speed * coastTau, so a move cuts power that many counts
// early. Learned from every stop (calibration first), so short and long moves both land.
float coastTau = 0;           // seconds
float velocity = 0, stopVelocity = 0;      // counts/s
long velCount = 0;
unsigned long velMs = 0;
long lastCount = 0;
unsigned long lastMotionMs = 0;
char settleFor = 0;           // 'M' move, 'H' home, 'C' calibration: what to do once coasting ends.
byte corrections = 0;
unsigned int homeStartPulses = 0;

void encoderEdge() {
  // Index = previous state * 4 + new state, state = A + 2*B. +1/-1 per valid step, 0 otherwise.
  static const int8_t STEP[16] = {0, 1, -1, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, -1, 1, 0};
  byte now = (PIND >> 2) & 0x03;           // D2 = A (bit 0), D3 = B (bit 1).
  if ((encState ^ now) == 0x03) ++missedEdges;
  encCount += STEP[(encState << 2) | now];
  encState = now;
}
ISR(PCINT2_vect) {                         // D4 = PCINT20, the index pulse.
  if (PIND & _BV(PIND4)) {
    indexCount = encCount;
    ++indexPulses;
  }
}
long readCount() {
  noInterrupts();
  long c = encCount;
  interrupts();
  return c;
}
float countsToDeg(long counts) {
  return (float)counts * 360.0f / (float)turnCounts;
}

void motorOff() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
  digitalWrite(REN, LOW);
  digitalWrite(LEN, LOW);
  mode = IDLE;
}
void motorOn(bool forward) {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
  digitalWrite(REN, HIGH);
  digitalWrite(LEN, HIGH);
  analogWrite(forward ? RPWM : LPWM, runPwm);
  started = millis();
  lastMotionMs = started;
  lastCount = readCount();
  velocity = 0;
  velCount = lastCount;
  velMs = started;
}
void fault(const __FlashStringHelper *why) {
  motorOff();
  setpointValid = false;
  Serial.print(F("ENCODER FAULT: "));
  Serial.println(why);
}
void stopAndSettle(char what, long progress) {
  motorOff();
  mode = SETTLING;
  settleFor = what;
  stopProgress = progress;
  stopVelocity = fabs(velocity);
  started = millis();
}
void clearInput() {
  inputLength = 0;
  inputOverflow = false;
  pending = 0;
}
bool requireIdle() {
  if (mode == IDLE) return true;
  Serial.println(F("Already running. Send S first."));
  return false;
}
void startCalibration() {
  if (!requireIdle()) return;
  fullTurnMs = 0;
  turnCounts = 0;
  coastTau = 0;
  setpointValid = false;
  calStartCount = readCount();
  mode = CALIBRATING;
  motorOn(true);
  Serial.println(F("Calibration running. Send M when your tile mark returns once."));
}
void finishCalibration() {
  if (mode != CALIBRATING) {
    Serial.println(F("Send C first to start calibration."));
    return;
  }
  unsigned long elapsed = millis() - started;
  long counts = readCount() - calStartCount;
  motorOff();
  if (elapsed < MIN_CAL_MS || elapsed > MAX_CAL_MS) {
    fullTurnMs = 0;
    Serial.println(F("Calibration rejected. Try again."));
    return;
  }
  fullTurnMs = elapsed;
  turnCounts = labs(counts) >= MIN_TURN_COUNTS ? counts : 0;
  if (turnCounts) {
    // Measure how far it coasts after power-off, so the first move already stops early.
    moveStartCount = calStartCount + counts;
    moveDir = turnCounts > 0 ? 1 : -1;
    stopAndSettle('C', 0);
  }
  Serial.print(F("One tile turn, motor-on time: "));
  Serial.print(fullTurnMs);
  Serial.print(F(" ms at PWM "));
  Serial.print(runPwm);
  Serial.print(F(", encoder: "));
  Serial.print(turnCounts);
  Serial.println(F(" counts. Now type an angle such as 60."));
  if (turnCounts == 0) Serial.println(F("No encoder counts seen: moves will be timed estimates."));
}
void setTurnTime(unsigned long ms) {
  if (!requireIdle()) return;
  if (ms < MIN_CAL_MS || ms > MAX_CAL_MS) {
    Serial.println(F("Turn time rejected: must be 500 to 60000 ms."));
    return;
  }
  fullTurnMs = ms;
  Serial.print(F("Turn time set: "));
  Serial.print(fullTurnMs);
  Serial.println(F(" ms."));
}
void setTurnCounts(long counts) {
  if (!requireIdle()) return;
  if (labs(counts) < MIN_TURN_COUNTS || labs(counts) > MAX_TURN_COUNTS) {
    Serial.println(F("Turn counts rejected: must be 100 to 10000000, either sign."));
    return;
  }
  turnCounts = counts;
  setpointValid = false;
  Serial.print(F("Turn counts set: "));
  Serial.print(turnCounts);
  Serial.println(F(" counts."));
}
void setPwm(unsigned long pwm) {
  if (!requireIdle()) return;
  if (pwm < 1 || pwm > 255) {
    Serial.println(F("PWM rejected: must be 1 to 255."));
    return;
  }
  // A turn time only holds at the speed it was timed at. Encoder counts hold at any speed.
  bool cleared = pwm != runPwm && fullTurnMs != 0;
  if (pwm != runPwm) fullTurnMs = 0;
  runPwm = (byte)pwm;
  Serial.print(F("Run PWM set: "));
  Serial.print(runPwm);
  Serial.println(cleared ? F(". Turn time cleared, recalibrate.") : F("."));
}
void printStatus() {
  Serial.print(F("Turn time: "));
  Serial.print(fullTurnMs);
  Serial.print(F(" ms, turn counts: "));
  Serial.print(turnCounts);
  Serial.print(F(", PWM: "));
  Serial.print(runPwm);
  Serial.print(F(", mode: "));
  switch (mode) {
    case IDLE: Serial.println(F("idle")); break;
    case CALIBRATING: Serial.println(F("calibrating")); break;
    case HOMING: Serial.println(F("homing")); break;
    case SETTLING: Serial.println(F("settling")); break;
    default: Serial.println(F("moving"));
  }
}
void printEncoder() {
  long count = readCount();
  noInterrupts();
  unsigned int pulses = indexPulses, missed = missedEdges;
  interrupts();
  Serial.print(F("Encoder: count "));
  Serial.print(count);
  Serial.print(F(", shaft "));
  Serial.print((float)count * 360.0f / (4.0f * ENC_PPR), 2);
  Serial.print(F(" deg, tile "));
  if (turnCounts) {
    Serial.print(countsToDeg(count - zeroCount), 2);
    Serial.print(F(" deg"));
  } else {
    Serial.print(F("? (calibrate)"));
  }
  Serial.print(homed ? F(", homed yes") : F(", homed no"));
  Serial.print(F(", index pulses "));
  Serial.print(pulses);
  Serial.print(F(", missed edges "));
  Serial.println(missed);
}
// Closed-loop move to an absolute encoder count.
void driveTo(long target) {
  setpoint = target;
  setpointValid = true;
  long distance = target - readCount();
  long minCounts = labs(turnCounts) / 720;   // Closer than half a degree: already there.
  if (labs(distance) <= minCounts) {
    Serial.print(F("Encoder move finished: at "));
    Serial.print(countsToDeg(readCount() - zeroCount), 2);
    Serial.println(F(" deg, already within 0.5 deg of the target."));
    return;
  }
  moveDir = distance > 0 ? 1 : -1;
  moveMagnitude = labs(distance);
  moveStartCount = readCount();
  mode = ENC_MOVING;
  motorOn((distance > 0) == (turnCounts > 0));
  Serial.print(F("Encoder move: to "));
  Serial.print(countsToDeg(target - zeroCount), 2);
  Serial.print(F(" deg, "));
  Serial.print(moveMagnitude);
  Serial.println(F(" counts."));
}
void startAngle(float degrees) {
  if (!requireIdle()) return;
  if (fullTurnMs == 0 && turnCounts == 0) {
    Serial.println(F("Calibrate first: mark tile, send C, then M at one full turn."));
    return;
  }
  if (isnan(degrees) || isinf(degrees) || fabs(degrees) < 1.0f || fabs(degrees) > 360.0f) {
    Serial.println(F("Enter an angle from 1 to 360, positive or negative."));
    return;
  }
  if (turnCounts) {
    corrections = 0;
    if (!setpointValid) setpoint = readCount();
    driveTo(setpoint + lround(degrees * (float)turnCounts / 360.0f));
    return;
  }
  moveMs = (unsigned long)(fabs(degrees) * (float)fullTurnMs / 360.0f + 0.5f);
  if (moveMs == 0 || moveMs > MAX_MOVE_MS) {
    Serial.println(F("Move time out of range."));
    return;
  }
  mode = MOVING;
  motorOn(degrees > 0.0f);
  Serial.print(F("Timed move estimate: "));
  Serial.print(degrees, 1);
  Serial.print(F(" deg, power on for "));
  Serial.print(moveMs);
  Serial.println(F(" ms."));
}
void startGoto(float degrees) {
  if (!requireIdle()) return;
  if (turnCounts == 0) {
    Serial.println(F("Goto needs the encoder: calibrate (C, M) with it connected."));
    return;
  }
  if (isnan(degrees) || isinf(degrees) || fabs(degrees) > 3600.0f) {
    Serial.println(F("Enter an angle from -3600 to 3600, such as G90."));
    return;
  }
  corrections = 0;
  driveTo(zeroCount + lround(degrees * (float)turnCounts / 360.0f));
}
void startHome() {
  if (!requireIdle()) return;
  noInterrupts();
  homeStartPulses = indexPulses;
  interrupts();
  calStartCount = readCount();
  setpointValid = false;
  mode = HOMING;
  motorOn(true);
  Serial.println(F("Homing: running forward to the encoder index pulse."));
}
void finishSettle() {
  long count = readCount();
  mode = IDLE;
  if (settleFor == 'H') {
    noInterrupts();
    zeroCount = indexCount;
    interrupts();
    homed = true;
    Serial.print(F("Homed at the index pulse. Coasted "));
    Serial.print(turnCounts ? countsToDeg(count - zeroCount) : 0.0f, 2);
    Serial.println(F(" deg past it."));
    return;
  }
  long progress = (count - moveStartCount) * moveDir;
  long coast = progress - stopProgress;
  if (coast >= 0 && stopVelocity >= 20.0f) {
    float tau = (float)coast / stopVelocity;
    coastTau = (coastTau > 0 && settleFor != 'C') ? (coastTau + tau) / 2 : tau;
  }
  if (settleFor == 'C') {
    Serial.print(F("Coast after power-off: "));
    Serial.print(coast);
    Serial.print(F(" counts ("));
    Serial.print(fabs(countsToDeg(coast)), 2);
    Serial.print(F(" deg) at full speed. Moves cut power early by speed x "));
    Serial.print((long)(coastTau * 1000.0f));
    Serial.println(F(" ms."));
    return;
  }
  if (labs(count - setpoint) > labs(turnCounts) / 720 && corrections < MAX_CORRECTIONS) {
    ++corrections;
    Serial.print(F("Correcting: off by "));
    Serial.print(countsToDeg(count - setpoint), 2);
    Serial.println(F(" deg."));
    driveTo(setpoint);
    return;
  }
  Serial.print(F("Encoder move finished: at "));
  Serial.print(countsToDeg(count - zeroCount), 2);
  Serial.print(F(" deg, target "));
  Serial.print(countsToDeg(setpoint - zeroCount), 2);
  Serial.print(F(" deg, error "));
  Serial.print(countsToDeg(count - setpoint), 2);
  Serial.println(F(" deg."));
}
void printPendingHelp() {
  if (pending == 'T') Serial.println(F("Type T followed by milliseconds, such as T4200."));
  else if (pending == 'K') Serial.println(F("Type K followed by counts, such as K20480."));
  else if (pending == 'G') Serial.println(F("Type G followed by degrees, such as G90."));
  else Serial.println(F("Type P followed by a PWM from 1 to 255, such as P80."));
}
void processNumber() {
  if (inputOverflow) {
    Serial.println(F("Command too long."));
  } else if (inputLength) {
    input[inputLength] = '\0';
    char *end;
    if (pending == 'T' || pending == 'P') {
      unsigned long value = strtoul(input, &end, 10);
      while (*end == ' ' || *end == '\t') ++end;
      if (end == input || *end != '\0') {
        printPendingHelp();
      } else if (pending == 'T') {
        setTurnTime(value);
      } else {
        setPwm(value);
      }
    } else if (pending == 'K') {
      long value = strtol(input, &end, 10);
      while (*end == ' ' || *end == '\t') ++end;
      if (end == input || *end != '\0') printPendingHelp();
      else setTurnCounts(value);
    } else {
      double degrees = strtod(input, &end);
      while (*end == ' ' || *end == '\t') ++end;
      if (end == input || *end != '\0') {
        if (pending) printPendingHelp();
        else Serial.println(F("Type C, M, S, H, E, G<deg>, T<ms>, K<counts>, P<pwm>, ?, or an angle such as 60."));
      } else if (pending == 'G') {
        startGoto((float)degrees);
      } else {
        startAngle((float)degrees);
      }
    }
  } else if (pending) {
    printPendingHelp();
  }
  clearInput();
}
void readCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    lastInputByte = millis();
    if (c == 's' || c == 'S') {
      motorOff();
      setpointValid = false;
      clearInput();
      while (Serial.available()) Serial.read();
      Serial.println(F("STOPPED. Motor disabled."));
      return;
    }
    char u = (c >= 'a' && c <= 'z') ? c - 'a' + 'A' : c;
    if (u == 'C') {
      clearInput();
      startCalibration();
    } else if (u == 'M') {
      clearInput();
      finishCalibration();
    } else if (u == 'H') {
      clearInput();
      startHome();
    } else if (u == 'E') {
      clearInput();
      printEncoder();
    } else if (u == 'T' || u == 'P' || u == 'K' || u == 'G') {
      clearInput();
      pending = u;
    } else if (c == '?') {
      clearInput();
      printStatus();
    } else if (c == '\n' || c == '\r') {
      if (inputLength || inputOverflow || pending) processNumber();
    } else if (inputLength < sizeof(input) - 1) {
      input[inputLength++] = c;
    } else {
      inputOverflow = true;
    }
  }
  // Accept numbers even with Serial Monitor set to "No line ending".
  if ((inputLength || inputOverflow || pending) && millis() - lastInputByte >= 150UL) processNumber();
}
void setup() {
  const byte outputs[] = {RPWM, LPWM, REN, LEN};
  for (byte i = 0; i < 4; ++i) {
    digitalWrite(outputs[i], LOW);
    pinMode(outputs[i], OUTPUT);
  }
  motorOff();
  // The AMT10E has push-pull CMOS outputs; the pull-ups only stop an unplugged encoder floating.
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  pinMode(ENC_X, INPUT_PULLUP);
  encState = (PIND >> 2) & 0x03;
  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderEdge, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), encoderEdge, CHANGE);
  PCMSK2 |= _BV(PCINT20);
  PCIFR |= _BV(PCIF2);
  PCICR |= _BV(PCIE2);
  Serial.begin(115200);
  Serial.println(F("T74 READY. Mark tile. C=start full turn, M=mark returned, S=stop, E=encoder."));
  Serial.println(F("After calibration type 60, 30, -30, etc. H=home to index, G90=go to 90 deg."));
}
void loop() {
  readCommands();
  unsigned long now = millis();
  unsigned long elapsed = now - started;
  long count = readCount();
  if (count != lastCount) {
    lastCount = count;
    lastMotionMs = now;
  }
  bool stalled = now - lastMotionMs >= STALL_MS;
  if (now - velMs >= VELOCITY_MS) {
    velocity = (float)(count - velCount) * 1000.0f / (float)(now - velMs);
    velCount = count;
    velMs = now;
  }
  if (mode == CALIBRATING && elapsed >= MAX_CAL_MS) {
    motorOff();
    fullTurnMs = 0;
    Serial.println(F("Calibration timed out; motor disabled."));
  } else if (mode == MOVING && elapsed >= moveMs) {
    motorOff();
    Serial.println(F("Timed move finished; motor disabled. Check tile mark after coasting."));
  } else if (mode == ENC_MOVING) {
    long progress = (count - moveStartCount) * moveDir;
    long remaining = moveMagnitude - progress;
    if (remaining <= 0 || remaining <= velocity * moveDir * coastTau) stopAndSettle('M', progress);
    else if (progress < -(labs(turnCounts) / 36)) fault(F("counting the wrong way. Recalibrate (C, M)."));
    else if (stalled && corrections) {
      // A short correction can be too little to start the motor at this PWM: accept where it is.
      corrections = MAX_CORRECTIONS;
      stopAndSettle('M', progress);
    }
    else if (stalled) fault(F("no counts for 1 s with the motor on. Check encoder wiring and that the motor turns."));
    else if (elapsed >= MAX_MOVE_MS) fault(F("move took over 60 s."));
  } else if (mode == HOMING) {
    noInterrupts();
    bool found = indexPulses != homeStartPulses;
    interrupts();
    if (found) stopAndSettle('H', 0);
    else if (stalled) fault(F("no counts for 1 s while homing. Check encoder wiring and that the motor turns."));
    else if (turnCounts && labs(count - calStartCount) > labs(turnCounts) * 5 / 4)
      fault(F("no index pulse in a full turn. Check the X wire on D4."));
    else if (elapsed >= MAX_CAL_MS) fault(F("homing took over 60 s."));
  } else if (mode == SETTLING && (now - lastMotionMs >= SETTLE_MS || elapsed >= MAX_SETTLE_MS)) {
    finishSettle();
  }
}

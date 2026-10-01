#include <Arduino.h>
#include <math.h>
#include <stdlib.h>

#include "t74.h"
#include "t74_control.h"

//  Pins come from config/t74.yaml via build_flash.sh (mtc/pins.py checks them).
#ifndef PIN_T74_RPWM
#error "PIN_T74_RPWM is not defined -- build with build_flash.sh, which takes the pins from config/t74.yaml"
#endif
#ifndef PIN_T74_LPWM
#error "PIN_T74_LPWM is not defined -- build with build_flash.sh, which takes the pins from config/t74.yaml"
#endif
#ifndef PIN_T74_R_EN
#error "PIN_T74_R_EN is not defined -- build with build_flash.sh, which takes the pins from config/t74.yaml"
#endif
#ifndef PIN_T74_L_EN
#error "PIN_T74_L_EN is not defined -- build with build_flash.sh, which takes the pins from config/t74.yaml"
#endif
#ifndef PIN_T74_ENC_A
#error "PIN_T74_ENC_A is not defined -- build with build_flash.sh, which takes the pins from config/t74.yaml"
#endif
#ifndef PIN_T74_ENC_B
#error "PIN_T74_ENC_B is not defined -- build with build_flash.sh, which takes the pins from config/t74.yaml"
#endif
#ifndef PIN_T74_ENC_X
#error "PIN_T74_ENC_X is not defined -- build with build_flash.sh, which takes the pins from config/t74.yaml"
#endif

/*  The T74 tile motor: PID position control on the encoder count, at 500 Hz from Timer1.
*
*   Wiring -- pins from config/t74.yaml; the defaults are t74-motor/'s:
*
*       IBT-2   RPWM D5 (forward)   LPWM D6 (reverse)   R_EN D7   L_EN D8   VCC 5V   GND GND
*       AMT10E2-V   A D2 (INT0)   B D3 (INT1)   X/index D4 (pin change)   5V   G GND
*
*   A and B need the Uno's two hardware interrupts (D2, D3): every edge of both channels is
*   counted ("4x decoding"). The index can be any pin; it takes its port's pin-change vector.
*   The encoder is on the MOTOR shaft, ~20.3 encoder turns per tile turn, at 500 PPR (DIP
*   switches 0 1 0 1) -- 5120 PPR is too fast for the Uno at speed.
*
*   The control law itself is in t74_control.h, shared with the simulation in test/. This file
*   is the hardware around it: the encoder, the timer, the bridge, the commands, the safety.
*   The board knows nothing about degrees or gains it was not sent: the host computes the gains
*   from the identified model and sends them on every connect.
*/

namespace t74
{
namespace
{
const byte RPWM = PIN_T74_RPWM, LPWM = PIN_T74_LPWM, REN = PIN_T74_R_EN, LEN = PIN_T74_L_EN;
const byte ENC_A = PIN_T74_ENC_A, ENC_B = PIN_T74_ENC_B, ENC_X = PIN_T74_ENC_X;
//  AMT10E DIP switch setting. MUST MATCH THE SWITCHES. Only used for the homing guard.
const unsigned int ENC_PPR = 500;

const unsigned int CONTROL_HZ = 500;
const float DT = 1.0f / CONTROL_HZ;
const unsigned long OPEN_LIMIT_TICKS = 60UL * CONTROL_HZ;   //  open-loop runs stop after 60 s
const unsigned int STALL_TICKS = CONTROL_HZ;                 //  saturated, no counts for 1 s
const unsigned int SETTLE_TICKS = CONTROL_HZ / 10;           //  in band for 100 ms = done
const unsigned long LINE_TIMEOUT_MS = 100;
const long MAX_IDENT_MS = 10000;
const uint8_t IDENT_EVERY = 2;                               //  a sample every 4 ms

//  ---- encoder, from interrupts ----
volatile long encCount = 0;
volatile long indexCount = 0;
volatile unsigned int indexPulses = 0;
volatile unsigned int missedEdges = 0;   //  both channels changed at once: too fast, or noise
volatile byte encState = 0;

//  the input registers and bit masks of A, B and X, looked up once in setup()
volatile uint8_t *aReg, *bReg, *xReg;
uint8_t aMask, bMask, xMask;

byte encoderState()
{
    return ((*aReg & aMask) ? 1 : 0) | ((*bReg & bMask) ? 2 : 0);   //  A + 2*B
}

void encoderEdge()
{
    //  Index = previous state * 4 + new state, state = A + 2*B. +1/-1 per valid step.
    static const int8_t STEP[16] = {0, 1, -1, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, -1, 1, 0};
    byte now = encoderState();
    if ((encState ^ now) == 0x03)
    {
        ++missedEdges;
    }
    encCount += STEP[(encState << 2) | now];
    encState = now;
}

long readCount()
{
    uint8_t s = SREG;
    cli();
    long c = encCount;
    SREG = s;
    return c;
}

//  ---- control state, shared by the 500 Hz interrupt and the main loop ----
enum Mode : uint8_t
{
    OFF,      //  motor disabled
    OPEN,     //  fixed PWM, no control: calibration runs
    HOLD,     //  closed loop: following the profile, or holding its goal
    HOMING,   //  closed loop at homeSpeed until the index pulse
    IDENT     //  open-loop PWM steps, sampled for the host to fit the model
};
volatile Mode mode = OFF;

t74ctl::Controller ctl;
t74ctl::Profile ref;             //  in raw encoder counts
t74ctl::Limits lim = {0, 0, 0};
float maxError = 0;              //  following error that means something is wrong, counts
float homeSpeed = 0;             //  counts/s, signed: the direction to search for the index
bool gainsSet = false, limitsSet = false;
volatile long zero = 0;          //  raw count that is position 0 (the interrupt sets it on homing)
volatile bool homed = false;

int openPwm = 0;                 //  OPEN, and the current IDENT step
int identPwm[2] = {0, 0};
unsigned long identTicks = 0;    //  ticks per IDENT step
unsigned long modeTicks = 0;     //  ticks since the current mode started
long stallCount = 0;
unsigned int stallTicks = 0;
unsigned int settledTicks = 0;
volatile bool awaitingDone = false;   //  a move/home is waiting to report "done"
//  Hold after a move (the default), or release the motor once it has settled on the target.
//  Set by the host with L; t74_hold toggles it.
bool holdAfterMove = true;
volatile bool releaseOnSettle = false;  //  this move/home/halt releases once settled
//  Released ON a goal by releaseOnSettle: a relative move still counts from that goal, not from
//  wherever the load has drifted to, so a run of moves stays on target.
volatile bool goalKept = false;
long homeStart = 0;
unsigned int homeStartPulses = 0;

//  events from the interrupt, reported by poll()
enum Event : uint8_t
{
    EV_DONE = 1,
    EV_HOMED = 2,
    EV_FAULT = 4,
    EV_IDENT_DONE = 8
};
volatile uint8_t events = 0;
volatile uint8_t faultCode = 0;
long homedAt = 0;

const char *const FAULTS[] = {
    "",
    "following error too large -- jammed, overloaded, or the gains have the wrong sign "
    "(re-run t74_identify)",
    "no encoder counts for 1 s at full PWM -- check the encoder wiring and that the motor turns",
    "no index pulse in 1.25 encoder turns -- check X on D4, and ENC_PPR",
    "open-loop run over 60 s",
};

//  identification samples: (ms since start, count), written by the interrupt
struct Sample
{
    uint16_t ms;
    long count;
};
const uint8_t RING = 32;
Sample ring[RING];
volatile uint8_t ringHead = 0, ringTail = 0;
volatile bool ringOverflow = false;

//  ---- the bridge ----
void driveRaw(int pwm)
{
    pwm = constrain(pwm, -255, 255);
    if (pwm >= 0)
    {
        analogWrite(LPWM, 0);
        analogWrite(RPWM, pwm);
    }
    else
    {
        analogWrite(RPWM, 0);
        analogWrite(LPWM, -pwm);
    }
}

void enable()
{
    digitalWrite(REN, HIGH);
    digitalWrite(LEN, HIGH);
}

void motorOff()
{
    analogWrite(RPWM, 0);
    analogWrite(LPWM, 0);
    digitalWrite(REN, LOW);
    digitalWrite(LEN, LOW);
    mode = OFF;
}

void fault(uint8_t code)
{
    motorOff();
    awaitingDone = false;
    releaseOnSettle = goalKept = false;
    faultCode = code;
    events |= EV_FAULT;
}

//  ---- the 500 Hz tick ----
void controlTick()
{
    long y = readCount();
    ++modeTicks;

    if (mode == OFF)
    {
        return;
    }
    if (mode == OPEN)
    {
        driveRaw(openPwm);
        if (modeTicks >= OPEN_LIMIT_TICKS)
        {
            fault(4);
        }
        return;
    }
    if (mode == IDENT)
    {
        unsigned long step = modeTicks / identTicks;
        if (step >= 2)
        {
            motorOff();
            events |= EV_IDENT_DONE;
            return;
        }
        driveRaw(identPwm[step]);
        if (modeTicks % IDENT_EVERY == 0)
        {
            uint8_t next = (ringHead + 1) % RING;
            if (next == ringTail)
            {
                ringOverflow = true;
            }
            else
            {
                ring[ringHead].ms = (uint16_t)(modeTicks * 1000UL / CONTROL_HZ);
                ring[ringHead].count = y;
                ringHead = next;
            }
        }
        return;
    }

    //  closed loop: HOLD or HOMING
    t74ctl::Limits now = lim;
    if (mode == HOMING)
    {
        noInterrupts();
        bool found = indexPulses != homeStartPulses;
        long at = indexCount;
        interrupts();
        if (found)
        {
            //  the index is the new zero: brake to it and hold
            zero = at;
            homed = true;
            homedAt = at;
            mode = HOLD;
            ref.moveTo((float)at);
            events |= EV_HOMED;
        }
        else if (labs(y - homeStart) > 5L * ENC_PPR)   //  1.25 encoder turns
        {
            fault(3);
            return;
        }
        else
        {
            now.vmax = fabsf(homeSpeed);
        }
    }

    ref.step(DT, now);
    float u = ctl.update(ref, lim, y, DT);
    float e = ref.pos - (float)y;

    if (fabsf(e) > maxError)
    {
        fault(1);
        return;
    }
    if (fabsf(u) >= 250 && y == stallCount)
    {
        if (++stallTicks >= STALL_TICKS)
        {
            fault(2);
            return;
        }
    }
    else
    {
        stallTicks = 0;
        stallCount = y;
    }
    if (mode == HOLD && !ref.active && fabsf(e) <= lim.band)
    {
        if (settledTicks < SETTLE_TICKS && ++settledTicks == SETTLE_TICKS)
        {
            if (awaitingDone)
            {
                awaitingDone = false;
                events |= EV_DONE;
            }
            if (releaseOnSettle)
            {
                releaseOnSettle = false;
                goalKept = true;
                motorOff();
                return;
            }
        }
    }
    else
    {
        settledTicks = 0;
    }
    driveRaw((int)lroundf(u));
}

//  ---- commands ----
/*  Reads `count` comma-separated numbers up to the end of the line. ALWAYS consumes through
*   the newline, so nothing left over is read as a command of its own.
*/
bool readNumbers(float *out, uint8_t count)
{
    char buf[64];
    uint8_t n = 0;
    bool ok = true;
    unsigned long start = millis();
    while (true)
    {
        if (millis() - start >= LINE_TIMEOUT_MS)
        {
            return false;
        }
        int c = Serial.read();
        if (c < 0)
        {
            continue;
        }
        if (c == '\n' || c == '\r')
        {
            break;
        }
        if (n < sizeof(buf) - 1)
        {
            buf[n++] = (char)c;
        }
        else
        {
            ok = false;
        }
    }
    buf[n] = '\0';
    if (!ok)
    {
        return false;
    }
    char *p = buf;
    for (uint8_t i = 0; i < count; ++i)
    {
        char *end;
        out[i] = (float)strtod(p, &end);
        if (end == p || isnan(out[i]) || isinf(out[i]))
        {
            return false;
        }
        p = end;
        if (i + 1 < count)
        {
            if (*p != ',')
            {
                return false;
            }
            ++p;
        }
    }
    return *p == '\0';
}

void printSamples()
{
    while (ringTail != ringHead)
    {
        Sample s = ring[ringTail];
        ringTail = (ringTail + 1) % RING;
        Serial.print(F("t74 id "));
        Serial.print(s.ms);
        Serial.print(' ');
        Serial.println(s.count);
    }
}

void reject(const __FlashStringHelper *why)
{
    Serial.print(F("t74 rejected: "));
    Serial.println(why);
}

long position()
{
    return readCount() - zero;
}

//  Start closed-loop control from where the motor is, if it was not already running.
void beginHold()
{
    if (mode != HOLD && mode != HOMING)
    {
        long y = readCount();
        noInterrupts();
        ctl.reset(y);
        ref.hold((float)y);
        stallTicks = settledTicks = 0;
        stallCount = y;
        modeTicks = 0;
        mode = HOLD;
        interrupts();
        enable();
    }
}

bool readyToMove()
{
    if (!gainsSet)
    {
        reject(F("no gains yet -- the host sends them; run t74_identify first"));
        return false;
    }
    if (!limitsSet)
    {
        reject(F("no limits yet -- the host sends them on connect"));
        return false;
    }
    return true;
}

void startMove(float goalRaw)
{
    beginHold();
    noInterrupts();
    ref.moveTo(goalRaw);
    settledTicks = 0;
    awaitingDone = true;
    releaseOnSettle = !holdAfterMove;
    goalKept = false;
    events &= ~EV_DONE;
    interrupts();
    Serial.print(F("t74 move from "));
    Serial.print(position());
    Serial.print(F(" to "));
    Serial.println((long)lroundf(goalRaw) - zero);
}

void report()
{
    noInterrupts();
    long y = encCount;
    float goal = ref.goal, pos = ref.pos, u = ctl.u;
    Mode m = mode;
    unsigned int pulses = indexPulses, missed = missedEdges;
    interrupts();
    static const char *const NAMES[] = {"off", "open", "hold", "homing", "identify"};
    Serial.print(F("t74 pos "));
    Serial.print(y - zero);
    Serial.print(F(" goal "));
    Serial.print((long)lroundf(goal) - zero);
    Serial.print(F(" err "));
    Serial.print(m == HOLD || m == HOMING ? (long)lroundf(pos) - y : 0L);
    Serial.print(F(" u "));
    Serial.print(m == OFF ? 0 : (int)lroundf(u));
    Serial.print(F(" mode "));
    Serial.print(NAMES[m]);
    Serial.print(F(" homed "));
    Serial.print(homed ? 1 : 0);
    Serial.print(F(" index "));
    Serial.print(pulses);
    Serial.print(F(" missed "));
    Serial.print(missed);
    Serial.print(F(" raw "));
    Serial.print(y);
    Serial.print(F(" hold "));
    Serial.println(holdAfterMove ? 1 : 0);
}
}

//  the index pulse: on whichever port's pin-change vector its pin belongs to
#if PIN_T74_ENC_X <= 7
#define T74_INDEX_VECT PCINT2_vect
#elif PIN_T74_ENC_X <= 13
#define T74_INDEX_VECT PCINT0_vect
#else
#define T74_INDEX_VECT PCINT1_vect
#endif
ISR(T74_INDEX_VECT)
{
    if (*xReg & xMask)
    {
        indexCount = encCount;
        ++indexPulses;
    }
}

//  Interrupts stay enabled inside the control tick, so encoder edges are never held up by it.
ISR(TIMER1_COMPA_vect, ISR_NOBLOCK)
{
    static volatile bool busy = false;
    if (busy)
    {
        return;   //  a tick overran the next one: skip rather than nest
    }
    busy = true;
    controlTick();
    busy = false;
}

void setup()
{
    const byte outputs[] = {RPWM, LPWM, REN, LEN};
    for (byte i = 0; i < 4; ++i)
    {
        digitalWrite(outputs[i], LOW);
        pinMode(outputs[i], OUTPUT);
    }
    motorOff();

    //  push-pull CMOS outputs; the pull-ups only stop an unplugged encoder floating
    pinMode(ENC_A, INPUT_PULLUP);
    pinMode(ENC_B, INPUT_PULLUP);
    pinMode(ENC_X, INPUT_PULLUP);
    aReg = portInputRegister(digitalPinToPort(ENC_A));
    bReg = portInputRegister(digitalPinToPort(ENC_B));
    xReg = portInputRegister(digitalPinToPort(ENC_X));
    aMask = digitalPinToBitMask(ENC_A);
    bMask = digitalPinToBitMask(ENC_B);
    xMask = digitalPinToBitMask(ENC_X);
    encState = encoderState();
    attachInterrupt(digitalPinToInterrupt(ENC_A), encoderEdge, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_B), encoderEdge, CHANGE);
    *digitalPinToPCMSK(ENC_X) |= _BV(digitalPinToPCMSKbit(ENC_X));
    PCIFR |= _BV(digitalPinToPCICRbit(ENC_X));
    PCICR |= _BV(digitalPinToPCICRbit(ENC_X));

    //  Timer1, CTC, /64: 250 kHz, compare every 500 counts = 500 Hz
    noInterrupts();
    TCCR1A = 0;
    TCCR1B = _BV(WGM12) | _BV(CS11) | _BV(CS10);
    OCR1A = (F_CPU / 64 / CONTROL_HZ) - 1;
    TCNT1 = 0;
    TIMSK1 = _BV(OCIE1A);
    interrupts();
}

bool handle(int c)
{
    float a[6];

    switch (c)
    {
    case 'J': //gains ('J' == just the numbers): kp,ki,kd,K,tau,friction
        if (!readNumbers(a, 6) || a[3] == 0 || a[4] <= 0 || a[5] < 0)
        {
            reject(F("J needs kp,ki,kd,K,tau,friction with K != 0, tau > 0, friction >= 0"));
            break;
        }
        noInterrupts();
        ctl.g = {a[0], a[1], a[2], a[3], a[4], a[5]};
        interrupts();
        gainsSet = true;
        Serial.println(F("t74 gains ok"));
        break;

    case 'L': //limits: vmax,amax,band,maxerror,homespeed (counts), hold after move (1/0)
        if (!readNumbers(a, 6) || a[0] <= 0 || a[1] <= 0 || a[2] < 0 || a[3] <= a[2] ||
            a[4] == 0 || (a[5] != 0 && a[5] != 1))
        {
            reject(F("L needs vmax,amax,band,maxerror,homespeed,hold: positive, maxerror > band, "
                     "hold 0 or 1"));
            break;
        }
        noInterrupts();
        lim = {a[0], a[1], a[2]};
        maxError = a[3];
        homeSpeed = a[4];
        holdAfterMove = a[5] != 0;
        interrupts();
        limitsSet = true;
        Serial.println(F("t74 limits ok"));
        break;

    case 'R': //relative move: from the current goal while holding, so errors do not add up
    case 'A': //absolute move, counts from zero; with a turn length, the shortest way round
        if (c == 'R' ? !readNumbers(a, 1) : !readNumbers(a, 2) || a[1] < 0)
        {
            reject(F("R<counts> or A<counts>,<counts per turn or 0>"));
            break;
        }
        if (!readyToMove())
        {
            break;
        }
        if (mode == OPEN || mode == IDENT)
        {
            reject(F("an open-loop run is going -- X first"));
            break;
        }
        {
            noInterrupts();
            float base = (mode == HOLD || goalKept) ? ref.goal : (float)encCount;
            interrupts();
            if (c == 'R')
            {
                startMove(roundf(base + a[0]));
                break;
            }
            float d = (float)zero + a[0] - base;
            if (a[1] > 0)
            {
                //  any whole number of turns lands on the same angle: take the nearest, so the
                //  move is at most half a turn
                d = fmodf(d, a[1]);
                if (d > a[1] / 2)
                {
                    d -= a[1];
                }
                else if (d <= -a[1] / 2)
                {
                    d += a[1];
                }
            }
            startMove(roundf(base + d));
        }
        break;

    case 'H': //home: closed loop at homeSpeed to the index pulse, which becomes zero
        if (!readyToMove())
        {
            break;
        }
        beginHold();
        noInterrupts();
        homeStart = readCount();
        homeStartPulses = indexPulses;
        ref.moveTo((float)homeStart + (homeSpeed > 0 ? 1 : -1) * 1e7f);   //  "keep going"
        settledTicks = 0;
        awaitingDone = true;
        releaseOnSettle = !holdAfterMove;
        goalKept = false;
        mode = HOMING;
        interrupts();
        Serial.println(F("t74 homing"));
        break;

    case 'Z': //zero here
        zero = readCount();
        homed = false;
        Serial.println(F("t74 zero"));
        break;

    case 'S': //halt: brake at amax to a stop, then hold there
        if (mode == HOLD || mode == HOMING)
        {
            noInterrupts();
            float stop = ref.vel * fabsf(ref.vel) / (2 * lim.amax);
            ref.moveTo(roundf(ref.pos + stop));
            mode = HOLD;
            awaitingDone = false;
            releaseOnSettle = !holdAfterMove;   //  with hold off, a halt releases once stopped
            interrupts();
        }
        else if (mode != OFF)
        {
            motorOff();
        }
        Serial.print(F("t74 halt pos "));
        Serial.println(position());
        break;

    case 'X': //release: motor off
        motorOff();
        awaitingDone = false;
        releaseOnSettle = goalKept = false;
        Serial.print(F("t74 off pos "));
        Serial.println(position());
        break;

    case 'O': //open-loop run at a fixed PWM until X (calibration)
        if (!readNumbers(a, 1) || fabsf(a[0]) > 255 || a[0] == 0)
        {
            reject(F("O needs a PWM, -255..255, not 0"));
            break;
        }
        noInterrupts();
        openPwm = (int)a[0];
        releaseOnSettle = goalKept = false;
        modeTicks = 0;
        mode = OPEN;
        interrupts();
        enable();
        Serial.print(F("t74 open "));
        Serial.print(openPwm);
        Serial.print(F(" pos "));
        Serial.println(position());
        break;

    case 'I': //identify: pwm1 for ms, then pwm2 for ms, sampled every 4 ms
        if (!readNumbers(a, 3) || fabsf(a[0]) > 255 || fabsf(a[1]) > 255 || a[2] < 100 ||
            a[2] > MAX_IDENT_MS)
        {
            reject(F("I needs pwm1,pwm2 (-255..255) and ms per step (100..10000)"));
            break;
        }
        noInterrupts();
        identPwm[0] = (int)a[0];
        identPwm[1] = (int)a[1];
        identTicks = (unsigned long)a[2] * CONTROL_HZ / 1000;
        ringHead = ringTail = 0;
        ringOverflow = false;
        releaseOnSettle = goalKept = false;
        modeTicks = 0;
        events &= ~EV_IDENT_DONE;
        mode = IDENT;
        interrupts();
        enable();
        Serial.print(F("t74 id start "));
        Serial.println(readCount());
        break;

    case 'E': //report
        report();
        break;

    default:
        return false;
    }
    return true;
}

void poll()
{
    printSamples();

    noInterrupts();
    uint8_t ev = events;
    events = 0;
    uint8_t code = faultCode;
    bool lost = ringOverflow;
    interrupts();

    if (ev & EV_FAULT)
    {
        Serial.print(F("t74 FAULT "));
        Serial.println(FAULTS[code]);
    }
    if (ev & EV_HOMED)
    {
        Serial.print(F("t74 homed at raw "));
        Serial.println(homedAt);
    }
    if (ev & EV_DONE)
    {
        noInterrupts();
        float goal = ref.goal;
        interrupts();
        Serial.print(F("t74 done pos "));
        Serial.print(position());
        Serial.print(F(" goal "));
        Serial.println((long)lroundf(goal) - zero);
    }
    if (ev & EV_IDENT_DONE)
    {
        printSamples();   //  any that landed after the drain above, so "done" really is last
        Serial.println(lost ? F("t74 id overflow -- samples were lost; try again")
                            : F("t74 id done"));
    }
}
}

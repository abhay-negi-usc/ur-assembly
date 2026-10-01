#pragma once

//  The screwdrive: 12 V DC motor on a Cytron MD10C R3, driven with sign-magnitude PWM + DIR.
//
//      'd<pct>\n'        signed speed in percent, -100..100, until told otherwise. 0 stops.
//      't<pct>,<ms>\n'   the same, for ms milliseconds, then stop
//      'p<duty>,<ms>\n'  raw PWM duty -255..255 for ms, then stop (the host's rpm command)
//      'a<p1>,<p2>,<ms>\n'  ramp linearly from p1% to p2% over ms, then HOLD p2%
//
//  A malformed line stops the motor. Any new command replaces a timed run in progress.
namespace screwdrive
{
void setup();
bool handle(int c);
void poll();     //  from loop(): services runs/ramps and reports "screwdrive run/ramp done"
void service();  //  from yield(): services runs/ramps only, never prints
}

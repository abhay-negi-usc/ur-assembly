#pragma once

//  The T74 tile motor: IBT-2 H-bridge + AMT10E2-V encoder, PID position control at 500 Hz.
//  All positions on the wire are encoder counts relative to the zero; the host converts degrees.
//
//      'J<kp>,<ki>,<kd>,<K>,<tau>,<friction>\n'   controller gains and model (from the host)
//      'L<vmax>,<amax>,<band>,<maxerr>,<homespeed>,<hold>\n'  limits (counts); hold 1 = hold
//                                                       after a move, 0 = release once settled
//      'R<counts>\n'  relative move      'A<counts>,<turn>\n'  absolute move (from zero);
//                                        turn > 0: shortest way round a turn of that many counts
//      'H'  home to the index pulse      'Z'  zero here
//      'S'  halt: brake to a stop and HOLD there      'X'  release: motor off
//      'O<pwm>\n'  open-loop run (calibration)   'I<pwm1>,<pwm2>,<ms>\n'  identify (step test)
//      'E'  report
//
//  It takes Timer1 over (the control interrupt) and needs both hardware-interrupt pins for its
//  encoder: build_flash.sh (mtc/pins.py) refuses combinations with modules that need them too.
namespace t74
{
void setup();
bool handle(int c);
void poll();     //  from loop(): prints move results, faults and identification samples
}

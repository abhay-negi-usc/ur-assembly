#pragma once

//  The toolchanger coupler: a servo that locks the ball bearings, and a proximity sensor that
//  checks a tool is really there.
//
//      '0'..'9'  hold (>0) / release (0)
//      's'       status, no retry
//      'r'       raw sensor reading, for calibration
//      'b'       toggle the sensor bypass
namespace coupler
{
void setup();
bool handle(int c);
void poll();
}

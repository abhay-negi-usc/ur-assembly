#include <Arduino.h>

#include "relay.h"

//  Pins come from config/relay.yaml via build_flash.sh (mtc/pins.py checks them).
#ifndef PIN_RELAY_K1
#error "PIN_RELAY_K1 is not defined -- build with build_flash.sh, which takes the pins from config/relay.yaml"
#endif

namespace relay
{
namespace
{
const int relayK1 = PIN_RELAY_K1; // powers on motor
bool motorActive = false;

void toggleMotorPower()
{
    if (motorActive)
    {
        analogWrite(relayK1, 0);
        motorActive = false;
        Serial.println("Motor Off");
    }
    else
    {
        analogWrite(relayK1, 201); //value corresponding to 4V (between required 2.5V and 5V to switch relay)
        motorActive = true;
        Serial.println("Motor On");
    }
}
}

void setup()
{
    pinMode(relayK1, OUTPUT);
    digitalWrite(relayK1, LOW);
}

bool handle(int c)
{
    if (c == 'm') //toggle motor ('m' == motor)
    {
        toggleMotorPower();
        return true;
    }
    return false;
}
}

#pragma once

/*  The T74's position control law, with no Arduino dependencies.
*
*   It is compiled into the firmware (t74.cpp runs it from a 500 Hz timer interrupt) AND into
*   firmware/test/t74_control_test.cpp, which runs it against a simulated motor + inertia on
*   the PC. So the maths that is tested is the maths that is flashed.
*
*   MODEL. Position y (encoder counts) responds to the PWM command u (-255..255) as a DC motor
*   driving an inertia with viscous damping:
*
*       tau * y'' + y' = K * u          i.e.   Y(s)/U(s) = K / (s (tau s + 1))
*
*   K is the steady speed per PWM step (counts/s per PWM, SIGNED: negative when forward PWM
*   counts down) and tau the mechanical time constant, which grows with the load's inertia.
*   Both are measured by `t74_identify`, which steps the PWM and fits this model.
*
*   The gearbox adds Coulomb friction: speed = K * (u - friction) once moving, and nothing at all
*   below it. `t74_identify` measures that too, from steps at two PWM levels.
*
*   CONTROL.  u = feedforward + friction + Kp e + Ki integral(e) + Kd e'    e = reference - y
*
*   The feedforward is the model's inverse applied to the motion profile, (v + tau a) / K, so
*   with a perfect model the error stays at zero and the feedback only has to correct the model's
*   mistakes and disturbances (an unbalanced load). Friction is cancelled by adding its PWM in
*   the direction of travel -- left to the integral instead, it makes the motor stick, wind up,
*   jump past the target and hunt. The feedback gains put all three closed-loop poles at -w
*   (mtc/modules/t74.py, pid_gains()):
*
*       tau s^3 + (1 + K Kd) s^2 + K Kp s + K Ki  =  tau (s + w)^3
*       Kp = 3 tau w^2 / K      Ki = tau w^3 / K      Kd = (3 tau w - 1) / K
*
*   -- a triple real pole: the error dies away critically damped, without ringing, AT THE LOAD
*   IT WAS IDENTIFIED WITH. Identify at the heaviest load. Away from it the poles move: heavier
*   loads ring more, and lighter ones tend to a damping ratio of sqrt(3)/2 = 0.87 (a fraction of
*   a percent of overshoot) -- the triple pole is the best-damped point, not a floor.
*/

#include <math.h>

namespace t74ctl
{

struct Gains
{
    float kp, ki, kd;       //  PWM per count, per count-second, per count/s
    float modelK;           //  counts/s per PWM, signed; 0 = not identified
    float modelTau;         //  seconds
    float friction;         //  PWM it takes to overcome Coulomb friction (>= 0)
};

struct Limits
{
    float vmax;             //  profile cruise speed, counts/s (> 0)
    float amax;             //  profile acceleration, counts/s^2 (> 0)
    float band;             //  "in position" half-width, counts: the integral freezes inside it
};

/*  Online trapezoidal profile: the reference accelerates at amax up to vmax, and brakes at amax
*   so that it stops exactly on the goal. Recomputed every tick, so the goal can change while
*   moving (a new move, or a halt) without a jump in the reference.
*/
struct Profile
{
    float pos = 0, vel = 0, acc = 0;   //  the reference, counts and counts/s and counts/s^2
    float goal = 0;
    bool active = false;

    void hold(float at)
    {
        pos = goal = at;
        vel = acc = 0;
        active = false;
    }

    void moveTo(float target)
    {
        goal = target;
        active = true;
    }

    void step(float dt, const Limits &lim)
    {
        float before = vel;
        if (active)
        {
            float dist = goal - pos;
            //  the fastest speed from which braking at amax still stops on the goal
            float reach = sqrtf(2.0f * lim.amax * fabsf(dist));
            float want = copysignf(fminf(lim.vmax, reach), dist);
            float dv = lim.amax * dt;
            vel += fmaxf(-dv, fminf(dv, want - vel));
            pos += vel * dt;
            if (fabsf(goal - pos) < 0.5f && fabsf(vel) <= dv)
            {
                pos = goal;
                vel = 0;
                active = false;
            }
        }
        acc = (vel - before) / dt;
    }
};

struct Controller
{
    Gains g = {0, 0, 0, 0, 0, 0};
    float integ = 0;      //  integral of the error, count-seconds
    float measVel = 0;    //  filtered measured speed, counts/s
    long lastY = 0;
    float u = 0;          //  last command

    //  The measured speed is filtered over a few ticks (a corner far above any sensible w),
    //  because a 2 ms difference of an integer count is coarse at low speed.
    static constexpr float VEL_ALPHA = 0.3f;

    void reset(long y)
    {
        integ = 0;
        measVel = 0;
        lastY = y;
        u = 0;
    }

    //  One control tick. Returns the PWM command, clamped to -255..255.
    float update(const Profile &r, const Limits &lim, long y, float dt)
    {
        float v = (float)(y - lastY) / dt;
        lastY = y;
        measVel += VEL_ALPHA * (v - measVel);

        float e = r.pos - (float)y;
        float ff = g.modelK != 0 ? (r.vel + g.modelTau * r.acc) / g.modelK : 0;
        float de = r.vel - measVel;

        //  Holding still and already in position: stop integrating, or static friction turns
        //  the integral into a slow hunt back and forth across the target.
        bool settled = !r.active && fabsf(e) <= lim.band;
        float next = settled ? integ : integ + e * dt;

        //  Friction compensation, in the direction the counts must go: the reference's while it
        //  moves, the error's while holding -- but not inside the band, where friction is what
        //  holds it still and pushing would only make it chatter.
        float dir = 0;
        if (r.active && r.vel != 0)
        {
            dir = r.vel > 0 ? 1.0f : -1.0f;
        }
        else if (!settled && fabsf(e) > lim.band)
        {
            dir = e > 0 ? 1.0f : -1.0f;
        }
        float fc = g.modelK != 0 ? dir * g.friction * (g.modelK > 0 ? 1.0f : -1.0f) : 0;

        float raw = ff + fc + g.kp * e + g.ki * next + g.kd * de;
        float out = fmaxf(-255.0f, fminf(255.0f, raw));
        //  Anti-windup: while saturated, refuse integration that pushes further into the limit.
        if (out != raw && g.ki * (next - integ) * raw > 0)
        {
            next = integ;
        }
        integ = next;
        u = out;
        return out;
    }
};

}

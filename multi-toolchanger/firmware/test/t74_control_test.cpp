/*  Simulates the T74 control law (../t74_control.h, the same code the firmware runs) against a
*   motor + inertia model, and checks the response is what the tuning promises.
*
*       g++ -O2 -std=c++11 -o /tmp/t74_test t74_control_test.cpp && /tmp/t74_test
*
*   (fake_board.py builds and runs it.) Exit status 0 means every check passed.
*
*   The plant is the identification model plus what the model leaves out -- integer encoder
*   counts, PWM saturation at 255, Coulomb friction, and an unbalanced load's constant torque:
*
*       tau * v' + v = K * (u - unbalance - friction(v))        y = floor(position)
*
*   integrated at 10 kHz, with the controller sampling it at 500 Hz and holding u in between.
*/

#include <math.h>
#include <stdio.h>

#include "../t74_control.h"

using t74ctl::Controller;
using t74ctl::Gains;
using t74ctl::Limits;
using t74ctl::Profile;

static const float CONTROL_DT = 1.0f / 500.0f;
static const int SUBSTEPS = 20;   //  plant steps per control tick: 10 kHz

//  Pole placement: all three closed-loop poles at -w (see t74_control.h). Same formula as
//  pid_gains() in mtc/modules/t74.py.
static Gains gainsFor(float K, float tau, float w, float friction)
{
    Gains g;
    g.friction = friction;
    g.kp = 3 * tau * w * w / K;
    g.ki = tau * w * w * w / K;
    g.kd = fmaxf(0.0f, 3 * tau * w - 1) / K;
    g.modelK = K;
    g.modelTau = tau;
    return g;
}

struct Plant
{
    float K, tau;
    float unbalance = 0;   //  constant load torque, as the PWM it takes to cancel it
    float friction = 0;    //  Coulomb friction, as PWM
    double pos = 0, vel = 0;

    long count() const { return (long)floor(pos); }

    void step(float u, float dt)
    {
        float drive = u - unbalance;
        if (vel == 0 && fabsf(drive) <= friction)
        {
            return;   //  stuck: static friction holds it
        }
        //  friction opposes the drive that makes this motion: in PWM terms, the sign of the
        //  velocity times the sign of K (forward PWM counts DOWN when K < 0)
        float dir = vel != 0 ? ((vel > 0) == (K > 0) ? 1.0f : -1.0f) : (drive > 0 ? 1.0f : -1.0f);
        double next = vel + dt * (K * (drive - friction * dir) - vel) / tau;
        //  friction stops it rather than reversing it
        vel = (friction > 0 && next * vel < 0) ? 0 : next;
        pos += vel * dt;
    }
};

struct Result
{
    float overshoot;   //  furthest past the goal, in the direction of travel (counts)
    float settle;      //  seconds from the profile finishing until |error| stays within band
    float finalErr;    //  counts
    float swing;       //  peak-to-peak position over the last second: hunting if > band
};

static Result move(Plant &p, Controller &c, Profile &r, const Limits &lim, float target,
                   float seconds)
{
    float start = r.goal;
    float dir = target >= start ? 1.0f : -1.0f;
    r.moveTo(target);
    Result res = {0, -1, 0, 0};
    float doneAt = -1, lastOut = -1;
    int ticks = (int)(seconds / CONTROL_DT);
    long lo = 0, hi = 0;
    bool late = false;
    for (int i = 0; i < ticks; ++i)
    {
        float t = i * CONTROL_DT;
        r.step(CONTROL_DT, lim);
        long y = p.count();
        float u = roundf(c.update(r, lim, y, CONTROL_DT));   //  the board writes whole PWM steps
        for (int k = 0; k < SUBSTEPS; ++k)
        {
            p.step(u, CONTROL_DT / SUBSTEPS);
        }
        float err = (float)p.count() - target;
        if (!r.active && doneAt < 0)
        {
            doneAt = t;
        }
        if (doneAt >= 0)
        {
            res.overshoot = fmaxf(res.overshoot, err * dir);
            if (fabsf(err) > lim.band)
            {
                lastOut = t;
            }
        }
        if (t > seconds - 1.0f)
        {
            long y2 = p.count();
            lo = late ? (y2 < lo ? y2 : lo) : y2;
            hi = late ? (y2 > hi ? y2 : hi) : y2;
            late = true;
        }
    }
    res.settle = doneAt < 0 ? -1 : fmaxf(0.0f, lastOut - doneAt);
    res.finalErr = (float)p.count() - target;
    res.swing = (float)(hi - lo);
    return res;
}

static int failures = 0;

static void check(bool ok, const char *what)
{
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok)
    {
        ++failures;
    }
}

static void report(const char *name, const Result &r)
{
    printf("%-12s overshoot %6.1f counts  settle %5.3f s  final %5.1f  swing %4.0f\n", name,
           r.overshoot, r.settle, r.finalErr, r.swing);
}

int main()
{
    //  Roughly the T74: 40,000 counts per tile turn, ~3.3 s per turn at PWM 100 -> ~120 counts/s
    //  per PWM. Negative, as calibrated (forward PWM counts down). tau is a plausible heaviest
    //  load; the light load is a third of it. Friction of 30 PWM: the motor "only hums" below ~45.
    const float K = -120.0f, TAU_MAX = 0.08f, W = 15.0f, FRICTION = 30.0f;
    const Limits lim = {9000.0f, 40000.0f, 5.0f};   //  ~81 deg/s, ~360 deg/s^2, ~0.05 deg

    struct Case
    {
        const char *name;
        float tau, unbalance, friction, compensation;
        float allowed;   //  overshoot allowed, counts; < 0 = report only
    };
    const float tight = lim.band + 2;   //  the band, plus a count each way of quantization
    const Case cases[] = {
        {"heaviest load (tuned for it)", TAU_MAX, 0, 0, 0, tight},
        {"lightest load (tau / 3)", TAU_MAX / 3, 0, 0, 0, 30},
        {"heaviest, unbalanced (40 PWM of torque)", TAU_MAX, 40, 0, 0, tight},
        {"heaviest, friction, compensated", TAU_MAX, 0, FRICTION, FRICTION, tight},
        {"heaviest, friction underestimated (2/3)", TAU_MAX, 0, FRICTION, FRICTION * 2 / 3, 30},
        {"heaviest, friction overestimated (4/3)", TAU_MAX, 0, FRICTION, FRICTION * 4 / 3, 30},
        {"lightest, friction + unbalance, compensated", TAU_MAX / 3, 40, FRICTION, FRICTION, 30},
        {"friction, NOT compensated (why it matters)", TAU_MAX, 0, FRICTION, 0, -1},
        {"heavier than tuned (2 x tau)", TAU_MAX * 2, 0, 0, 0, -1},
    };
    const float moves[] = {9000, 4500, 4600, -2000};   //  +81, -40, +1, -59 deg (absolute)

    for (const Case &cs : cases)
    {
        Gains g = gainsFor(K, TAU_MAX, W, cs.compensation);
        Plant p;
        p.K = K;
        p.tau = cs.tau;
        p.unbalance = cs.unbalance;
        p.friction = cs.friction;
        Controller c;
        c.g = g;
        c.reset(p.count());
        Profile r;
        r.hold(0);
        printf("%s%s\n", cs.name, cs.allowed < 0 ? "  -- info only" : "");
        float worst = 0;
        bool ended = true, steady = true;
        for (float target : moves)
        {
            char label[64];
            snprintf(label, sizeof label, "  to %+.0f", target);
            Result res = move(p, c, r, lim, target, 3.0f);
            report(label, res);
            worst = fmaxf(worst, res.overshoot);
            ended = ended && fabsf(res.finalErr) <= lim.band + 1;
            steady = steady && res.swing <= lim.band;
        }
        if (cs.allowed >= 0)
        {
            char what[160];
            snprintf(what, sizeof what, "%s: overshoot %.0f <= %.0f counts, ends in position, "
                     "no hunting", cs.name, worst, cs.allowed);
            check(worst <= cs.allowed && ended && steady, what);
        }
    }

    {
        Gains g = gainsFor(K, TAU_MAX, W, 0);
        printf("\ngains for K %.0f, tau %.3f s, w %.0f rad/s: kp %.4f ki %.4f kd %.5f\n", K,
               TAU_MAX, W, g.kp, g.ki, g.kd);
    }

    //  The disturbance response is where the pole placement shows directly: a sudden unbalance
    //  while holding should be pushed back without crossing to the other side.
    {
        Plant p;
        p.K = K;
        p.tau = TAU_MAX;
        Controller c;
        c.g = gainsFor(K, TAU_MAX, W, 0);
        c.reset(0);
        Profile r;
        r.hold(0);
        float peak = 0, cross = 0;
        for (int i = 0; i < 2000; ++i)
        {
            if (i == 250)
            {
                p.unbalance = 60;   //  load shifts while holding
            }
            r.step(CONTROL_DT, lim);
            float u = roundf(c.update(r, lim, p.count(), CONTROL_DT));
            for (int k = 0; k < SUBSTEPS; ++k)
            {
                p.step(u, CONTROL_DT / SUBSTEPS);
            }
            float e = (float)p.count();
            peak = fmaxf(peak, fabsf(e));
            if (i > 250 && e * K > 0)   //  pushed by the unbalance one way; past zero the other
            {
                cross = fmaxf(cross, fabsf(e));
            }
        }
        printf("  load shift while holding: pushed %.0f counts off, recovered with %.0f counts "
               "past the target, ends %ld\n", peak, cross, p.count());
        check(cross <= lim.band, "load shift: recovers without ringing past the target");
        check(labs(p.count()) <= (long)lim.band + 1, "load shift: back in position");
    }

    printf("\n%s\n", failures ? "SOME CHECKS FAILED" : "ALL CONTROL CHECKS PASSED");
    return failures ? 1 : 0;
}

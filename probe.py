"""Phase 0 probe for the SO-101 arm on the LeRobot stack.

Offline (default): load the calibration from LeRobot's cache, print the typed
observation/action schema, and convert the arm's last known tick pose to degrees.

--hw: read-only bus probe. Never calls robot.connect(): its configure() step disables
torque on all six motors (the arm drops) and rewrites ~10 registers per motor, and a
calibration mismatch sends it into calibrate(), which overwrites the motors' Homing_Offset.
Here the port is opened, registers are only read, and the port is closed with torque as-is.
"""

import argparse
import sys

from lerobot.motors import MotorNormMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from serial.tools import list_ports

ROBOT_ID = "exp23_follower"  # the classmate's original LeRobot id for this arm
# Arm pose when artist's teleop died on 2026-10-02 (repos/opencode/artist.md).
LAST_KNOWN_TICKS = {1: 1854, 2: 2336, 3: 2046, 4: 2873, 5: 3311, 6: 866}
READ_RETRIES = 2


def make_robot(port: str) -> SO101Follower:
    # Also covers Robot.__del__, which calls disconnect() on garbage collection.
    config = SO101FollowerConfig(port=port, id=ROBOT_ID, disable_torque_on_disconnect=False)
    return SO101Follower(config)


def unit(robot: SO101Follower, motor: str) -> str:
    return "deg" if robot.bus.motors[motor].norm_mode is MotorNormMode.DEGREES else "%"


def offline(robot: SO101Follower) -> bool:
    if not robot.calibration:
        print(f"FAIL: no calibration at {robot.calibration_fpath}")
        return False
    print(f"calibration          {robot.calibration_fpath}")
    print(f"observation_features {robot.observation_features}")
    print(f"action_features      {robot.action_features}")

    values = robot.bus._normalize(LAST_KNOWN_TICKS)
    back = robot.bus._unnormalize(values)
    print("\nlast known pose (offline conversion)")
    print(f"{'motor':<14}{'id':>3}{'range':>12}{'zero@':>8}{'tick':>6}{'value':>9} unit {'back':>5}")
    worst = 0
    for motor, m in robot.bus.motors.items():
        cal = robot.calibration[motor]
        is_deg = unit(robot, motor) == "deg"
        zero = (cal.range_min + cal.range_max) / 2 if is_deg else cal.range_min
        worst = max(worst, abs(back[m.id] - LAST_KNOWN_TICKS[m.id]))
        print(
            f"{motor:<14}{m.id:>3}{f'{cal.range_min}-{cal.range_max}':>12}{zero:>8g}"
            f"{LAST_KNOWN_TICKS[m.id]:>6}{values[m.id]:>9.2f} {unit(robot, motor):<4} {back[m.id]:>5}"
        )
    ok = worst <= 1
    print(f"tick -> value -> tick round trip: max error {worst} tick -> {'OK' if ok else 'FAIL'}")
    return ok


def hw(robot: SO101Follower) -> bool:
    try:
        robot.bus.connect()  # open port, ping ids 1-6, read firmware versions
    except ConnectionError as e:
        print(f"FAIL: {e}\navailable ports: {[p.device for p in list_ports.comports()]}")
        return False
    try:
        eeprom = robot.bus.read_calibration()
        print(f"\n{'motor':<14}{'motor EEPROM off/min/max':>26}{'file off/min/max':>22}  match")
        same = True
        for motor, e in eeprom.items():
            f = robot.calibration[motor]
            on_motor = (e.homing_offset, e.range_min, e.range_max)
            in_file = (f.homing_offset, f.range_min, f.range_max)
            same &= on_motor == in_file
            print(f"{motor:<14}{str(on_motor):>26}{str(in_file):>22}  {on_motor == in_file}")
        if not same:
            print("WARNING: robot.connect() would enter calibrate() and offer to overwrite motor EEPROM.")

        obs = robot.get_observation()
        raw = robot.bus.sync_read("Present_Position", normalize=False, num_retry=READ_RETRIES)
        torque = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=READ_RETRIES)
        temp = robot.bus.sync_read("Present_Temperature", normalize=False, num_retry=READ_RETRIES)
        print(f"\n{'motor':<14}{'tick':>6}{'value':>9} unit torque tempC")
        for motor in robot.bus.motors:
            print(
                f"{motor:<14}{raw[motor]:>6}{obs[f'{motor}.pos']:>9.2f} {unit(robot, motor):<4}"
                f"{torque[motor]:>7}{temp[motor]:>6}"
            )
        return same
    finally:
        robot.bus.disconnect(disable_torque=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 LeRobot probe (offline by default).")
    parser.add_argument("--port", default="COM11")
    parser.add_argument("--hw", action="store_true", help="read-only bus probe; the port must be free")
    args = parser.parse_args()

    robot = make_robot(args.port)
    ok = offline(robot)
    if ok and args.hw:
        ok = hw(robot)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

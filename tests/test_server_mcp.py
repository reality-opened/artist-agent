"""End to end through the MCP protocol, as an agent sees it: spawn `server.py --sim` over stdio and drive it.
Run from the repo root (~2-3 min): .venv\\Scripts\\python.exe -m tests.test_server_mcp
"""

import asyncio
import base64
import hashlib
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
TOOLS = {"arm_status", "move_joints", "move_to", "move_by", "set_gripper", "grasp", "stop", "release_torque", "park",
         "look", "find_objects", "locate_point", "find_marker", "servo_to", "calibrate_camera", "calibrate_rig",
         "auto_calibration"}
REAL_RIG = ROOT / "calibration" / "rig.json"


def text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


def data(result) -> dict:
    assert not result.is_error, text(result)
    out = result.structured_content
    return out.get("result", out) if isinstance(out, dict) and set(out) == {"result"} else out


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


async def scenario() -> None:
    rig_before = digest(REAL_RIG)
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server.py"), "--sim"], cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name: t for t in (await session.list_tools()).tools}
            assert TOOLS <= set(tools), set(tools)
            assert all(tools[n].description for n in TOOLS)
            assert tools["move_to"].output_schema and "status" in str(tools["move_to"].output_schema)

            st = data(await session.call_tool("arm_status", {}))
            assert st["calibration_ok"] and st["busy"] is False and "simulation" in st["rig_calibration"], st
            assert st["session_reset"] == "session start: parked (reached)", st["session_reset"]
            assert st["tcp_above_bench_mm"] > 100, st["tcp_above_bench_mm"]

            refused = await session.call_tool("release_torque", {"which": "all"})
            assert refused.is_error and "confirm" in text(refused), text(refused)
            refused = await session.call_tool("move_joints", {"targets": {"shoulder_lift": -20}})
            assert refused.is_error and "outside" in text(refused), text(refused)
            refused = await session.call_tool("move_by", {"dx_mm": 0, "dy_mm": 0, "dz_mm": -200})
            assert refused.is_error and "bench" in text(refused), text(refused)

            found = data(await session.call_tool("find_objects", {}))
            assert found["arm_excluded"] and found["frame_note"] is None, found
            cans = [o for o in found["objects"] if o["kind"] == "round_top" and abs(o["radius_mm"] - 33) < 4]
            assert len(cans) == 1, found["objects"]
            can = cans[0]
            assert abs(can["top_center_base_mm"][0] - 300) < 4 and abs(can["top_center_base_mm"][1] - 60) < 4, can
            assert abs(can["top_center_base_mm"][2] - 122) < 4 and abs(can["approach_base_mm"][2] - 182) < 4, can
            print(f"  find_objects: {len(found['objects'])} object(s), can top at {can['top_center_base_mm']} mm (base)")

            shot = await session.call_tool("look", {"grid": True})
            img = shot.content[0]
            assert img.type == "image" and img.mime_type == "image/jpeg"
            assert base64.b64decode(img.data)[:2] == b"\xff\xd8"

            u, v = can["image_px"]
            point = data(await session.call_tool("locate_point", {"u_px": u, "v_px": v}))
            assert point["source"] == "depth" and point["point_base_mm"] is not None, point
            assert abs(point["point_base_mm"][0] - 300) < 40 and abs(point["point_base_mm"][1] - 60) < 40, point

            x, y, z = can["approach_base_mm"]
            moved = data(await session.call_tool("move_to", {"x_mm": x, "y_mm": y, "z_mm": z}))
            print(f"  move_to approach: {moved['status']}, TCP {moved['tcp_error_mm']} mm from it")
            assert moved["status"] in ("reached", "off_target") and moved["tcp_error_mm"] < 15, moved

            marker = data(await session.call_tool("find_marker", {}))
            assert marker["confirmed_by_motion"] and marker["marker_base_mm"] is not None, marker
            xc, yc, zc = can["approach_cam_mm"]
            servo = data(await session.call_tool("servo_to", {"x_cam_mm": xc, "y_cam_mm": yc, "z_cam_mm": zc}))
            print(f"  servo_to: {servo['status']}, {servo['error_mm']} mm after {servo['probes']} probes + "
                  f"{servo['iterations']} steps ({servo['duration_s']} s)")
            assert servo["status"] in ("reached", "resolution_floor") and servo["error_mm"] <= 8, servo

            pan = data(await session.call_tool("arm_status", {}))["joints_deg"]["shoulder_pan"]
            slow = asyncio.create_task(session.call_tool("move_joints", {"targets": {"shoulder_pan": pan - 40},
                                                                         "speed_deg_s": 8}))
            await asyncio.sleep(1.5)
            busy = await session.call_tool("move_by", {"dx_mm": 0, "dy_mm": 0, "dz_mm": 5})
            assert busy.is_error and "another motion" in text(busy), text(busy)
            stopped = data(await session.call_tool("stop", {}))
            moved = data(await slow)
            assert stopped["status"] == "held" and moved["status"] == "stopped", (stopped, moved)
            print(f"  stop mid-move: pan stopped after {moved['final']['shoulder_pan'] - pan:+.1f} deg of -40")

            for name in ("calibrate_camera", "calibrate_rig"):
                refused = await session.call_tool(name, {})
                assert refused.is_error and "confirm=true" in text(refused), text(refused)
            data(await session.call_tool("set_gripper", {"opening_pct": 90}))
            cal = data(await session.call_tool("calibrate_camera", {"confirm": True}))
            print(f"  calibrate_camera: {cal['poses']} poses, rms {cal['rms_mm']} mm -> {cal['saved']}")
            grip = data(await session.call_tool("arm_status", {}))["gripper_opening_pct"]
            assert abs(grip - 20) < 5, f"calibrate_camera left the gripper at {grip} %"
            # bench height pinned: the 10 mm ball's front-surface bias along the bench normal stays in the residual
            assert cal["status"] == "calibrated" and cal["rms_mm"] < 5.0 and "artist-agent-sim-rig" in cal["saved"], cal
            again = data(await session.call_tool("find_objects", {}))
            can2 = [o for o in again["objects"] if o["kind"] == "round_top"][0]
            shift = max(abs(a - b) for a, b in zip(can2["top_center_base_mm"], can["top_center_base_mm"]))
            print(f"  after recalibration the can moved {shift:.1f} mm in the base frame")
            # the marker is measured on its front surface: that constant camera-frame bias ends up in the camera pose
            assert shift < 10, (shift, can2)

            grasp = data(await session.call_tool("grasp", {}))
            assert grasp["held"] is False and "nothing" in grasp["reason"], grasp
            park = data(await session.call_tool("park", {}))
            assert park["status"] in ("reached", "settled"), park
    assert digest(REAL_RIG) == rig_before, "a simulation run modified the real calibration/rig.json"


def test_mcp_session_in_sim() -> None:
    asyncio.run(scenario())


def test_session_start_parks_once() -> None:
    import argparse

    import server
    from arm import JOINTS

    rig = server.Rig(argparse.Namespace(sim=True, gemini_serial="SIM"))
    arm = rig.arm
    rest = dict(arm.rest_pose_deg)
    left = arm.move_joints({"shoulder_pan": rest["shoulder_pan"] + 20, "elbow_flex": rest["elbow_flex"] - 15,
                            "gripper": 80}, speed_dps=40)  # where a previous session might have left it
    assert left.status in ("reached", "settled"), left
    rig.session_reset()
    st = arm.state()
    assert all(abs(st.joints_deg[m] - rest[m]) < 1.5 for m in JOINTS), (st.joints_deg, rest)
    assert abs(st.gripper_opening_pct - server.RESET_GRIPPER_PCT) < 5, st.gripper_opening_pct  # set_gripper: 2 deg
    assert rig.reset_note == "session start: parked (reached), gripper closed to 20 % (reached)", rig.reset_note
    arm.move_joints({"shoulder_pan": rest["shoulder_pan"] + 10}, speed_dps=40)
    rig.session_reset()  # once per server process: later calls leave the arm where the agent put it
    assert abs(arm.state().joints_deg["shoulder_pan"] - rest["shoulder_pan"] - 10) < 1.5
    arm.disconnect()


if __name__ == "__main__":
    test_session_start_parks_once()
    print("ok test_session_start_parks_once")
    test_mcp_session_in_sim()
    print("ok test_mcp_session_in_sim")

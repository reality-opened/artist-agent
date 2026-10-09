"""Hardware acceptance through the real MCP server, the way an agent drives it. Needs the arm, the Gemini, and a clear
space of ~10 cm around the gripper; nothing else may own the serial port (artist's teleop, another MCP server).

    .venv\\Scripts\\python.exe -m tests.hardware_check                      # automatic checks (~3 min)
    .venv\\Scripts\\python.exe -m tests.hardware_check --hand-block         # + a person blocks a joint by hand
    .venv\\Scripts\\python.exe -m tests.hardware_check --recalibrate-camera # + camera recalibration (rewrites rig.json)

Every check prints PASS/FAIL with its numbers; images and a JSON report go to tests/hardware_runs/<time>/.
"""

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
from mcp import ClientSession, StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent


class Run:
    def __init__(self, out: Path):
        self.out, self.results = out, []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.results.append(dict(check=name, ok=bool(ok), detail=detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {detail}", flush=True)
        return ok


def text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


async def call(session, name, **args):
    result = await session.call_tool(name, args)
    if result.is_error:
        return None, text(result)
    out = result.structured_content
    return (out.get("result", out) if isinstance(out, dict) and set(out) == {"result"} else out), None


async def scenario(run: Run, opts) -> None:
    server = [str(ROOT / "server.py"), "--port", opts.port]
    params = StdioServerParameters(command=sys.executable, args=server, cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            st, err = await call(s, "arm_status")
            if st is None:
                run.check("arm_status", False, err)
                return
            hot = max(st["temperature_c"].values())
            safe = (st["calibration_ok"] and not st["thermal_lockout"] and hot < 45 and st["tcp_above_bench_mm"] > 60)
            run.check("arm_status / preconditions", safe,
                      f"calibration_ok={st['calibration_ok']} max {hot} C, TCP {st['tcp_above_bench_mm']} mm above the bench,"
                      f" rig: {st['rig_calibration']}")
            if not safe:
                print("  stopping: preconditions not met (needs calibration_ok, < 45 C, gripper > 60 mm above the bench)")
                return

            shot = await s.call_tool("look", {"grid": True})
            ok = not shot.is_error and shot.content and shot.content[0].type == "image"
            if ok:
                (run.out / "look_grid.jpg").write_bytes(base64.b64decode(shot.content[0].data))
            run.check("look(grid)", ok, "saved look_grid.jpg" if ok else text(shot))

            objs, err = await call(s, "find_objects")
            run.check("find_objects", objs is not None and objs["arm_excluded"],
                      err or f"{len(objs['objects'])} object(s), arm masked out: {objs['arm_excluded']}, "
                             f"note: {objs['frame_note']}")
            if objs:
                for o in objs["objects"][:5]:
                    print(f"        #{o['id']} {o['kind']:<9} h {o['height_mm']:5.1f} mm  {o['length_mm']:.0f}x{o['width_mm']:.0f} mm"
                          f"  top(base) {o['top_center_base_mm']}  px {o['image_px']}")
                (run.out / "objects.json").write_text(json.dumps(objs, indent=1))
                shot = await s.call_tool("look", {})
                if not shot.is_error:
                    (run.out / "look_objects.jpg").write_bytes(base64.b64decode(shot.content[0].data))

            pt, err = await call(s, "locate_point", u_px=640, v_px=500)
            run.check("locate_point(640, 500)", pt is not None,
                      err or f"{pt['source']}: base {pt['point_base_mm']} mm, {pt['height_above_bench_mm']} mm above bench")

            mk, err = await call(s, "find_marker")
            st, _ = await call(s, "arm_status")
            if mk is not None and mk["marker_base_mm"] is not None:
                gap = float(np.linalg.norm(np.subtract(mk["marker_base_mm"], st["tcp_mm"])))
                run.check("find_marker vs FK TCP", mk["confirmed_by_motion"] and gap < 25,
                          f"marker {mk['marker_base_mm']} vs model TCP {st['tcp_mm']}: {gap:.1f} mm apart "
                          f"({mk['candidates']} same-colour blob(s))")
            else:
                run.check("find_marker", False, err or f"no base-frame point: {mk and mk['frame_note']}")

            for d in ((20, 0, 0), (-20, 0, 0), (0, 20, 0), (0, -20, 0), (0, 0, 20), (0, 0, -20)):
                r, err = await call(s, "move_by", dx_mm=d[0], dy_mm=d[1], dz_mm=d[2], speed_deg_s=12)
                ok = r is not None and r["status"] in ("reached", "off_target") and r["tcp_error_mm"] < 10
                run.check(f"move_by {d}", ok, err or f"{r['status']}, TCP {r['tcp_error_mm']} mm off, joints "
                          f"{max(abs(v) for v in r['move']['error'].values()):.2f} deg, {r['move']['mean_cycle_ms']} ms/cycle")

            if mk is not None:
                target = np.add(mk["marker_cam_mm"], [25.0, 0.0, 0.0]).tolist()
                sv, err = await call(s, "servo_to", x_cam_mm=target[0], y_cam_mm=target[1], z_cam_mm=target[2])
                ok = sv is not None and sv["status"] in ("reached", "resolution_floor") and sv["error_mm"] <= 8
                run.check("servo_to marker +25 mm (camera x)", ok, err or f"{sv['status']}, {sv['error_mm']} mm after "
                          f"{sv['probes']} probes + {sv['iterations']} steps, {sv['duration_s']} s, J sv {sv['singular_values']}")
                if sv is not None:
                    back, _ = await call(s, "servo_to", x_cam_mm=mk["marker_cam_mm"][0], y_cam_mm=mk["marker_cam_mm"][1],
                                         z_cam_mm=mk["marker_cam_mm"][2])
                    run.check("servo_to back (reuses J)", back is not None and back["probes"] == 0 and
                              back["status"] in ("reached", "resolution_floor"),
                              "" if back is None else f"{back['status']}, {back['error_mm']} mm, {back['duration_s']} s")

            st, _ = await call(s, "arm_status")
            low = await s.call_tool("move_by", {"dx_mm": 0, "dy_mm": 0, "dz_mm": -(st["tcp_above_bench_mm"] + 20)})
            st2, _ = await call(s, "arm_status")
            still = max(abs(st2["joints_deg"][m] - st["joints_deg"][m]) for m in st["joints_deg"])
            run.check("bench refusal (no motion)", low.is_error and "bench" in text(low) and still < 0.5,
                      f"{text(low)[:90]} | moved {still:.2f} deg")

            pan = st2["joints_deg"]["shoulder_pan"]
            task = asyncio.create_task(call(s, "move_joints", targets={"shoulder_pan": pan + 15}, speed_deg_s=5))
            await asyncio.sleep(1.2)
            busy, berr = await call(s, "move_by", dx_mm=0, dy_mm=0, dz_mm=5)
            stop, _ = await call(s, "stop")
            mv, _ = await task
            await asyncio.sleep(0.8)
            st3, _ = await call(s, "arm_status")
            coast = abs(st3["joints_deg"]["shoulder_pan"] - mv["final"]["shoulder_pan"]) if mv else 99
            run.check("busy + stop mid-move", busy is None and "another motion" in (berr or "") and mv is not None and
                      mv["status"] == "stopped" and coast < 3,
                      f"stop {stop}, moved {mv and mv['final']['shoulder_pan'] - pan:+.1f} deg of +15, coasted {coast:.2f} deg")
            r, _ = await call(s, "move_joints", targets={"shoulder_pan": pan}, speed_deg_s=10)

            r, err = await call(s, "set_gripper", opening_pct=60)
            run.check("set_gripper 60%", r is not None and r["status"] in ("reached", "settled"),
                      err or f"{r['status']}, opening {r['final'].get('gripper')} %")
            g, err = await call(s, "grasp")
            run.check("grasp on empty jaws", g is not None and not g["held"] and "nothing" in g["reason"],
                      err or f"held={g['held']}: {g['reason']} (opening {g['opening_pct']} %)")
            await call(s, "set_gripper", opening_pct=20)

            if opts.hand_block:
                print("\n  HAND BLOCK: the shoulder pan will turn +25 deg slowly. Put a hand against the arm to stop it,"
                      " hold 2 s, then let go.")
                input("  press Enter when ready... ")
                st, _ = await call(s, "arm_status")
                pan = st["joints_deg"]["shoulder_pan"]
                mv, err = await call(s, "move_joints", targets={"shoulder_pan": pan + 25}, speed_deg_s=6)
                held = mv["final"]["shoulder_pan"] if mv else None
                await asyncio.sleep(1.5)
                st2, _ = await call(s, "arm_status")
                lunge = abs(st2["joints_deg"]["shoulder_pan"] - held) if held is not None else 99
                run.check("hand block: blocked, no lunge", mv is not None and mv["status"] == "blocked" and lunge < 3,
                          err or f"{mv['status']} ({mv['blocked']}), moved after release {lunge:.2f} deg")
                await call(s, "move_joints", targets={"shoulder_pan": pan}, speed_deg_s=10)

            if opts.recalibrate_camera:
                before, _ = await call(s, "arm_status")
                cal, err = await call(s, "calibrate_camera", confirm=True)
                run.check("calibrate_camera", cal is not None and cal["rms_mm"] < 8,
                          err or f"{cal['poses']} poses, rms {cal['rms_mm']} mm -> {cal['saved']}")

            pk, err = await call(s, "park")
            st, _ = await call(s, "arm_status")
            run.check("park", pk is not None and pk["status"] == "reached",
                      err or f"{pk['status']}, worst joint {max(abs(v) for v in pk['error'].values()):.2f} deg, "
                             f"TCP {st['tcp_above_bench_mm']} mm above the bench")
            run.check("temperatures after the run", max(st["temperature_c"].values()) < 50,
                      f"{st['temperature_c']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--hand-block", action="store_true", help="interactive: a person blocks the arm by hand")
    ap.add_argument("--recalibrate-camera", action="store_true", help="also re-fit the camera pose (writes rig.json)")
    opts = ap.parse_args()
    out = ROOT / "tests" / "hardware_runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    run = Run(out)
    t0 = time.monotonic()
    asyncio.run(scenario(run, opts))
    passed = sum(r["ok"] for r in run.results)
    (out / "report.json").write_text(json.dumps(dict(results=run.results, seconds=round(time.monotonic() - t0)), indent=1))
    print(f"\n{passed}/{len(run.results)} checks passed in {time.monotonic() - t0:.0f} s -> {out}")
    return 0 if passed == len(run.results) and run.results else 1


if __name__ == "__main__":
    sys.exit(main())

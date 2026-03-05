import os
import re
import signal
import subprocess
import sys
import time


def terminate_process_group(process: subprocess.Popen, timeout_seconds: float = 3.0) -> None:
    if process.poll() is not None:
        return

    process_group_id = os.getpgid(process.pid)
    os.killpg(process_group_id, signal.SIGTERM)

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)

    if process.poll() is None:
        os.killpg(process_group_id, signal.SIGKILL)


def get_listening_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["ss", "-ltnp"],
        capture_output=True,
        text=True,
        check=False,
    )

    pids: set[int] = set()
    for line in result.stdout.splitlines():
        if f":{port}" not in line:
            continue
        pids.update(int(match) for match in re.findall(r"pid=(\d+)", line))
    return sorted(pids)


def kill_stale_port_listeners(port: int) -> None:
    stale_pids = get_listening_pids(port)
    if not stale_pids:
        return

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    time.sleep(0.5)

    remaining_pids = get_listening_pids(port)
    for pid in remaining_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if remaining_pids:
        time.sleep(0.2)


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))
    kill_stale_port_listeners(port)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--reload",
        "--port",
        str(port),
    ]

    process = subprocess.Popen(command, start_new_session=True)

    try:
        return process.wait()
    except KeyboardInterrupt:
        terminate_process_group(process)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
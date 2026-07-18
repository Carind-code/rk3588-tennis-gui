# -*- coding: utf-8 -*-
"""
Process Manager — launch and control backend Python / shell scripts.

Provides non-blocking subprocess management with stdout/stderr capture
emitted as Qt signals for real-time console display.
"""

import os
import signal
import subprocess
import sys
import threading

from PyQt5.QtCore import QObject, pyqtSignal


def _read_stream(stream, callback, stop_event):
    """Read lines from a stream in a daemon thread and call callback for each."""
    try:
        for raw_line in iter(stream.readline, b""):
            if stop_event.is_set():
                break
            try:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
            except Exception:
                line = str(raw_line)
            if line:
                callback(line)
    except (ValueError, OSError):
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


class ProcessManager(QObject):
    """Manage subprocess lifecycle with Qt signal integration.

    Typical usage::

        pm = ProcessManager()
        pm.signal_stdout.connect(my_console.append)
        pm.start_script("match_judgement", "bash start_test.sh", cwd="...")
        ...
        pm.stop_process("match_judgement")
    """

    # Emitted when a process writes to stdout / stderr
    signal_stdout = pyqtSignal(str)
    signal_stderr = pyqtSignal(str)

    # Emitted on lifecycle changes: (process_name, returncode)
    signal_process_started = pyqtSignal(str)
    signal_process_finished = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._processes: dict = {}       # name -> Popen
        self._threads: dict = {}          # name -> list[Thread]
        self._stop_events: dict = {}      # name -> threading.Event

    # ── Public API ────────────────────────────────────────────────

    def start_script(self, name: str, cmd: list, cwd: str = None):
        """Launch a subprocess.

        Args:
            name: Unique identifier for this process (e.g. "match_judgement").
            cmd: Command + arguments as a list (e.g. ["bash", "start_test.sh"]).
            cwd: Working directory for the subprocess.

        If a process with the same *name* is already running, it is stopped first.
        """
        if self.is_running(name):
            self.stop_process(name)

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # Ensure PATH includes common locations on RK3588
        env.setdefault("PATH", "/usr/bin:/usr/local/bin:/bin")

        stop_event = threading.Event()
        self._stop_events[name] = stop_event

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
        self._processes[name] = proc

        # Reader threads
        threads = []
        for stream, signal_obj in [
            (proc.stdout, self.signal_stdout),
            (proc.stderr, self.signal_stderr),
        ]:
            t = threading.Thread(
                target=_read_stream,
                args=(stream, signal_obj.emit, stop_event),
                daemon=True,
            )
            t.start()
            threads.append(t)
        self._threads[name] = threads

        self.signal_process_started.emit(name)

        # Monitor thread for completion
        threading.Thread(
            target=self._monitor_completion,
            args=(name, proc),
            daemon=True,
        ).start()

        return proc

    def stop_process(self, name: str):
        """Gracefully terminate a process and its reader threads.

        Strategy (Linux):
        1. Send SIGTERM to parent only → lets it clean up children
        2. Wait up to 6s for graceful exit
        3. If still alive, force-kill the process group
        """
        proc = self._processes.pop(name, None)
        stop_event = self._stop_events.pop(name, None)

        if proc is None:
            return

        # Signal reader threads to stop
        if stop_event:
            stop_event.set()

        # Step 1: Gentle SIGTERM to parent only (not pg)
        try:
            proc.terminate()  # SIGTERM to parent process only
        except (ProcessLookupError, OSError):
            pass

        # Step 2: Wait for graceful cleanup
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            # Step 3: Escalate — force kill process group
            try:
                if sys.platform == "win32":
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

        # Wait for reader threads
        for t in self._threads.pop(name, []):
            t.join(timeout=2)

    def stop_all(self):
        """Stop all managed processes."""
        for name in list(self._processes.keys()):
            self.stop_process(name)

    def is_running(self, name: str) -> bool:
        """Check if a named process is still running."""
        proc = self._processes.get(name)
        if proc is None:
            return False
        return proc.poll() is None

    def get_process(self, name: str):
        """Return the underlying Popen object, or None."""
        return self._processes.get(name)

    # ── Internal ──────────────────────────────────────────────────

    def _monitor_completion(self, name: str, proc: subprocess.Popen):
        """Watch for process exit and emit finished signal."""
        try:
            proc.wait()
        except Exception:
            pass
        finally:
            returncode = proc.returncode if proc.returncode is not None else -1
            self.signal_process_finished.emit(name, returncode)
            # Cleanup
            self._processes.pop(name, None)
            self._stop_events.pop(name, None)
            self._threads.pop(name, None)

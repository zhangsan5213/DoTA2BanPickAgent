"""TensorBoard logging utilities."""

import os
import subprocess
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Manages TensorBoard logging."""

    def __init__(self, log_dir: str, enabled: bool = True, port: int = 6006):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether to enable logging
            port: Port for TensorBoard server
        """
        if not log_dir.startswith("runs"):
            log_dir = os.path.join("runs", os.path.basename(log_dir))
        self.log_dir = log_dir
        self.enabled = enabled
        self.port = port
        self.writer: SummaryWriter = None
        self.tb_process = None

    def start(self) -> SummaryWriter:
        """Start TensorBoard logging.

        Returns:
            SummaryWriter instance or None if disabled
        """
        if not self.enabled:
            return None

        # Start TensorBoard process
        self.tb_process = self._start_tensorboard_process(self.log_dir, self.port)

        # Create writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"[+] TensorBoard writer initialized")
        print(f"[+] Log directory: {self.log_dir}")

        return self.writer

    def _start_tensorboard_process(self, log_dir: str, port: int):
        """Start TensorBoard process in background."""
        cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port)]
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[+] TensorBoard started at http://localhost:{port}")
            return process
        except FileNotFoundError:
            print(
                "[!] TensorBoard not found. Make sure it's installed: pip install tensorboard"
            )
            return None

    def close(self):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            print("[+] TensorBoard writer closed")

        if self.tb_process is not None:
            print(f"[+] TensorBoard process is running in background")
            print(f"[+] You can view logs at http://localhost:{self.port}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

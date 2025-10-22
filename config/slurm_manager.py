"""
SLURM job queue manager with GPU capacity limits per node.
"""

import subprocess
import time
import re
from typing import Dict, Optional


class SlurmGPUManager:
    """Manages SLURM job submission with per-node GPU limits."""

    def __init__(self, node_limits: Optional[Dict[str, int]] = None):
        """
        Initialize manager with GPU limits per node.

        Args:
            node_limits: Dict mapping node names to max concurrent GPU jobs.
                        Default: {"voyager": 2, "laniakea": 4}
        """
        self.node_limits = node_limits or {"voyager": 2, "laniakea": 4}

    def get_running_jobs_per_node(self) -> Dict[str, int]:
        """
        Query SLURM to get number of running GPU jobs per node.

        Returns:
            Dict mapping node names to number of running jobs
        """
        try:
            # Query SLURM for running jobs with node info
            cmd = [
                "squeue",
                "-u",
                subprocess.getoutput("whoami"),
                "-h",
                "-o",
                "%N",
                "-t",
                "RUNNING",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                print(f"Warning: squeue command failed: {result.stderr}")
                return {}

            # Count jobs per node
            node_counts = {}
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                # Extract node name (handle formats like "voyager" or "laniakea[1-2]")
                node_match = re.match(r"(\w+)", line)
                if node_match:
                    node = node_match.group(1)
                    node_counts[node] = node_counts.get(node, 0) + 1

            return node_counts

        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"Warning: Failed to query SLURM status: {e}")
            return {}

    def get_available_node(self, preferred_nodes: list = None) -> Optional[str]:
        """
        Find a node with available GPU capacity.

        Args:
            preferred_nodes: List of nodes in order of preference

        Returns:
            Node name with available capacity, or None if all are full
        """
        if preferred_nodes is None:
            preferred_nodes = list(self.node_limits.keys())

        running_jobs = self.get_running_jobs_per_node()

        for node in preferred_nodes:
            if node not in self.node_limits:
                continue

            current_jobs = running_jobs.get(node, 0)
            limit = self.node_limits[node]

            if current_jobs < limit:
                print(f"✓ Node {node}: {current_jobs}/{limit} GPUs in use - Available")
                return node
            else:
                print(f"✗ Node {node}: {current_jobs}/{limit} GPUs in use - Full")

        return None

    def wait_for_available_node(
        self,
        preferred_nodes: list = None,
        check_interval: int = 30,
        max_wait: int = 3600,
    ) -> Optional[str]:
        """
        Wait until a node has available GPU capacity.

        Args:
            preferred_nodes: List of nodes in order of preference
            check_interval: Seconds between status checks
            max_wait: Maximum seconds to wait (default: 1 hour)

        Returns:
            Node name with available capacity, or None if timeout
        """
        if preferred_nodes is None:
            preferred_nodes = list(self.node_limits.keys())

        elapsed = 0
        print(f"\n{'=' * 60}")
        print(f"Waiting for available GPU on nodes: {preferred_nodes}")
        print(f"Checking every {check_interval}s (max wait: {max_wait}s)")
        print(f"{'=' * 60}\n")

        while elapsed < max_wait:
            node = self.get_available_node(preferred_nodes)
            if node:
                return node

            print(
                f"All preferred nodes are full. Waiting {check_interval}s... "
                f"(elapsed: {elapsed}s/{max_wait}s)"
            )
            time.sleep(check_interval)
            elapsed += check_interval

        print(f"Warning: Timeout after {max_wait}s waiting for available node")
        return None


# Global manager instance
_gpu_manager = None


def get_gpu_manager(node_limits: Optional[Dict[str, int]] = None) -> SlurmGPUManager:
    """Get or create the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = SlurmGPUManager(node_limits)
    return _gpu_manager

"""
SLURM job queue manager with GPU capacity limits per node.
"""

import subprocess
import time
import re
from typing import Dict, Optional


class SlurmGPUManager:
    """Manages SLURM job submission with per-node GPU limits."""

    def __init__(
        self,
        node_limits: Optional[Dict[str, int]] = None,
        node_capacity: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize manager with GPU limits per node.

        Args:
            node_limits: Dict mapping node names to max concurrent GPU jobs for this user.
                        Default: {"voyager": 3, "laniakea": 6}
            node_capacity: Dict mapping node names to total GPU capacity.
                        Default: {"voyager": 4, "laniakea": 8}
        """
        self.node_limits = node_limits or {"voyager": 3, "laniakea": 6}
        self.node_capacity = node_capacity or {"voyager": 4, "laniakea": 8}

    def get_running_jobs_per_node(self) -> Dict[str, int]:
        """
        Query SLURM to get number of running GPU jobs per node for current user.

        Returns:
            Dict mapping node names to number of running jobs for current user
        """
        user_jobs, _ = self._get_job_counts()
        return user_jobs

    def get_total_running_jobs_per_node(self) -> Dict[str, int]:
        """
        Query SLURM to get total number of running GPU jobs per node (all users).

        Returns:
            Dict mapping node names to total number of running jobs
        """
        _, total_jobs = self._get_job_counts()
        return total_jobs

    def _get_job_counts(self) -> tuple[Dict[str, int], Dict[str, int]]:
        """
        Single SLURM query to get both user and total job counts per node.

        Returns:
            Tuple of (user_jobs_dict, total_jobs_dict)
        """
        try:
            current_user = subprocess.getoutput("whoami")

            # Single query for all running jobs with user info
            cmd = [
                "squeue",
                "-h",
                "-o",
                "%N %u",  # Node and user
                "-t",
                "RUNNING",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                print(f"Warning: squeue command failed: {result.stderr}")
                return {}, {}

            # Count jobs per node
            user_counts = {}
            total_counts = {}

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                node_str, user = parts[0], parts[1]

                # Extract node name (handle formats like "voyager" or "laniakea[1-2]")
                node_match = re.match(r"(\w+)", node_str)
                if node_match:
                    node = node_match.group(1)
                    total_counts[node] = total_counts.get(node, 0) + 1
                    if user == current_user:
                        user_counts[node] = user_counts.get(node, 0) + 1

            return user_counts, total_counts

        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"Warning: Failed to query SLURM status: {e}")
            return {}, {}

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

        # Single SLURM query for both user and total jobs
        user_running_jobs, total_running_jobs = self._get_job_counts()

        for node in preferred_nodes:
            if node not in self.node_limits:
                continue

            user_jobs = user_running_jobs.get(node, 0)
            user_limit = self.node_limits[node]
            total_jobs = total_running_jobs.get(node, 0)
            total_capacity = self.node_capacity.get(node, user_limit)

            # Check both user limit and total node capacity
            if user_jobs < user_limit and total_jobs < total_capacity:
                print(
                    f"✓ Node {node}: User {user_jobs}/{user_limit}, Total {total_jobs}/{total_capacity} GPUs - Available"
                )
                return node
            else:
                reason = []
                if user_jobs >= user_limit:
                    reason.append(f"user limit reached ({user_jobs}/{user_limit})")
                if total_jobs >= total_capacity:
                    reason.append(f"node full ({total_jobs}/{total_capacity})")
                print(f"✗ Node {node}: {', '.join(reason)}")

        return None

    def wait_for_available_node(
        self,
        preferred_nodes: list = None,
        check_interval: int = 30,
        max_wait: int = 14400,  # 3600,
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


def get_gpu_manager(
    node_limits: Optional[Dict[str, int]] = None,
    node_capacity: Optional[Dict[str, int]] = None,
) -> SlurmGPUManager:
    """Get or create the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = SlurmGPUManager(node_limits, node_capacity)
    return _gpu_manager

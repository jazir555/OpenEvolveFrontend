"""
Resource Monitoring Utility

Monitors system resources during load testing including CPU, memory,
disk I/O, and network usage.

Usage:
    python monitor_resources.py --duration 300 --output resources.json
"""

import asyncio
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import deque

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource monitoring disabled")


class ResourceMonitor:
    """
    Monitor system resources during load testing.

    Tracks:
    - CPU usage (percentage)
    - Memory usage (RSS, virtual)
    - Disk I/O (read/write bytes)
    - Network I/O (sent/received bytes)
    - Connection counts
    """

    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize resource monitor.

        Args:
            sampling_interval: Time between samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.samples = deque()
        self.is_running = False
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None

    async def start(self, duration: int):
        """
        Start monitoring for specified duration.

        Args:
            duration: Monitoring duration in seconds
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("Cannot start monitoring - psutil not available")
            return

        self.is_running = True
        self.samples.clear()

        # Get initial network I/O counters
        net_io = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()

        logger.info(f"Starting resource monitoring for {duration} seconds")
        logger.info(f"Sampling interval: {self.sampling_interval}s")

        start_time = time.time()
        sample_num = 0

        while self.is_running and (time.time() - start_time) < duration:
            try:
                sample = self._collect_sample(sample_num, start_time)
                self.samples.append(sample)
                sample_num += 1

                await asyncio.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error collecting sample: {e}")

        self.is_running = False
        logger.info(f"Monitoring complete. Collected {len(self.samples)} samples")

    def _collect_sample(self, sample_num: int, start_time: float) -> Dict:
        """
        Collect a single resource sample.

        Args:
            sample_num: Sample number
            start_time: Test start time

        Returns:
            Dictionary with resource metrics
        """
        if not self.process:
            return {}

        sample = {
            "sample_num": sample_num,
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_time": time.time() - start_time
        }

        # CPU metrics
        sample["cpu_percent"] = self.process.cpu_percent(interval=0.1)
        sample["cpu_count"] = psutil.cpu_count()

        # Memory metrics
        mem_info = self.process.memory_info()
        sample["memory_rss_mb"] = mem_info.rss / (1024 * 1024)
        sample["memory_vms_mb"] = mem_info.vms / (1024 * 1024)

        # System memory
        sys_mem = psutil.virtual_memory()
        sample["system_memory_percent"] = sys_mem.percent
        sample["system_memory_available_gb"] = sys_mem.available / (1024**3)

        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                sample["disk_read_mb"] = disk_io.read_bytes / (1024 * 1024)
                sample["disk_write_mb"] = disk_io.write_bytes / (1024 * 1024)
                sample["disk_read_count"] = disk_io.read_count
                sample["disk_write_count"] = disk_io.write_count
        except Exception:
            pass

        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                sample["net_sent_mb"] = net_io.bytes_sent / (1024 * 1024)
                sample["net_recv_mb"] = net_io.bytes_recv / (1024 * 1024)
                sample["net_packets_sent"] = net_io.packets_sent
                sample["net_packets_recv"] = net_io.packets_recv
        except Exception:
            pass

        # Connections
        try:
            connections = self.process.connections()
            sample["connection_count"] = len(connections)
        except Exception:
            sample["connection_count"] = 0

        # Thread count
        try:
            sample["thread_count"] = self.process.num_threads()
        except Exception:
            sample["thread_count"] = 0

        # Open files
        try:
            sample["open_files"] = len(self.process.open_files())
        except Exception:
            sample["open_files"] = 0

        return sample

    def stop(self):
        """Stop monitoring."""
        self.is_running = False

    def get_samples(self) -> List[Dict]:
        """
        Get all collected samples.

        Returns:
            List of sample dictionaries
        """
        return list(self.samples)

    def get_summary(self) -> Dict:
        """
        Get summary statistics of collected samples.

        Returns:
            Dictionary with summary statistics
        """
        if not self.samples:
            return {}

        summary = {
            "total_samples": len(self.samples),
            "duration_seconds": self.samples[-1]["elapsed_time"] if self.samples else 0
        }

        # CPU statistics
        cpu_values = [s["cpu_percent"] for s in self.samples]
        summary["cpu"] = {
            "avg": sum(cpu_values) / len(cpu_values),
            "max": max(cpu_values),
            "min": min(cpu_values)
        }

        # Memory statistics
        mem_values = [s["memory_rss_mb"] for s in self.samples]
        summary["memory"] = {
            "avg_mb": sum(mem_values) / len(mem_values),
            "max_mb": max(mem_values),
            "min_mb": min(mem_values),
            "growth_mb": mem_values[-1] - mem_values[0] if len(mem_values) > 1 else 0
        }

        # Connection statistics
        conn_values = [s["connection_count"] for s in self.samples]
        summary["connections"] = {
            "avg": sum(conn_values) / len(conn_values),
            "max": max(conn_values),
            "min": min(conn_values)
        }

        return summary

    def save_samples(self, output_path: str):
        """
        Save samples to JSON file.

        Args:
            output_path: Path to save samples
        """
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "sampling_interval": self.sampling_interval,
            "summary": self.get_summary(),
            "samples": list(self.samples)
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Samples saved to {output_path}")

    def detect_anomalies(self) -> List[Dict]:
        """
        Detect anomalies in resource usage.

        Returns:
            List of anomaly descriptions
        """
        anomalies = []

        if len(self.samples) < 10:
            return anomalies

        # Check for memory leaks (continuous growth)
        mem_values = [s["memory_rss_mb"] for s in self.samples]
        first_quarter = mem_values[:len(mem_values)//4]
        last_quarter = mem_values[-len(mem_values)//4:]

        if first_quarter and last_quarter:
            avg_first = sum(first_quarter) / len(first_quarter)
            avg_last = sum(last_quarter) / len(last_quarter)
            growth = (avg_last - avg_first) / avg_first if avg_first > 0 else 0

            if growth > 0.5:  # 50% growth threshold
                anomalies.append({
                    "type": "memory_leak",
                    "severity": "HIGH" if growth > 1.0 else "MEDIUM",
                    "description": f"Memory growth of {growth:.1%} detected",
                    "details": f"Average memory increased from {avg_first:.1f} MB to {avg_last:.1f} MB"
                })

        # Check for high CPU usage
        cpu_values = [s["cpu_percent"] for s in self.samples]
        avg_cpu = sum(cpu_values) / len(cpu_values)

        if avg_cpu > 80:
            anomalies.append({
                "type": "high_cpu",
                "severity": "HIGH",
                "description": f"High average CPU usage: {avg_cpu:.1f}%",
                "details": f"Maximum CPU usage: {max(cpu_values):.1f}%"
            })

        # Check for connection leaks
        conn_values = [s["connection_count"] for s in self.samples]
        if len(conn_values) > 1:
            conn_growth = conn_values[-1] - conn_values[0]
            if conn_growth > 50:  # 50 connection threshold
                anomalies.append({
                    "type": "connection_leak",
                    "severity": "MEDIUM",
                    "description": f"Connection count increased by {conn_growth}",
                    "details": f"Start: {conn_values[0]}, End: {conn_values[-1]}"
                })

        return anomalies


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor system resources during load testing"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Monitoring duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--output",
        default="resource_monitoring.json",
        help="Output file for samples (default: resource_monitoring.json)"
    )
    parser.add_argument(
        "--detect-anomalies",
        action="store_true",
        help="Detect and report anomalies"
    )

    args = parser.parse_args()

    if not PSUTIL_AVAILABLE:
        print("ERROR: psutil is required for resource monitoring")
        print("Install with: pip install psutil")
        return 1

    # Create monitor
    monitor = ResourceMonitor(sampling_interval=args.interval)

    # Start monitoring
    await monitor.start(args.duration)

    # Save samples
    monitor.save_samples(args.output)

    # Print summary
    summary = monitor.get_summary()
    print("\n" + "="*60)
    print("RESOURCE MONITORING SUMMARY")
    print("="*60)
    print(f"Duration: {summary['duration_seconds']:.1f} seconds")
    print(f"Samples: {summary['total_samples']}")
    print(f"\nCPU Usage:")
    print(f"  Average: {summary['cpu']['avg']:.1f}%")
    print(f"  Max: {summary['cpu']['max']:.1f}%")
    print(f"  Min: {summary['cpu']['min']:.1f}%")
    print(f"\nMemory Usage:")
    print(f"  Average: {summary['memory']['avg_mb']:.1f} MB")
    print(f"  Max: {summary['memory']['max_mb']:.1f} MB")
    print(f"  Growth: {summary['memory']['growth_mb']:.1f} MB")
    print(f"\nConnections:")
    print(f"  Average: {summary['connections']['avg']:.1f}")
    print(f"  Max: {summary['connections']['max']}")

    # Detect anomalies
    if args.detect_anomalies:
        anomalies = monitor.detect_anomalies()
        if anomalies:
            print(f"\n{'='*60}")
            print("ANOMALIES DETECTED")
            print("="*60)
            for anomaly in anomalies:
                print(f"\n[{anomaly['severity']}] {anomaly['type']}")
                print(f"  {anomaly['description']}")
                print(f"  {anomaly['details']}")
        else:
            print(f"\n{'='*60}")
            print("No anomalies detected")
            print("="*60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

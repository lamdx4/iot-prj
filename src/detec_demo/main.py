#!/usr/bin/env python3
"""
Main Entry Point - Start Victim and Detector Servers
Orchestrates multiprocess communication via shared queue
"""

import argparse
import logging
import os
import sys
import signal
import socket
from multiprocessing import Process, Queue

# Add parent directory to path so we can import victim and detector packages
sys.path.insert(0, os.path.dirname(__file__))

from victim.victim import victim_server_main
from detector.detector import detector_server_main


# -----------------------
# Setup Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='[MAIN] [%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# -----------------------
# Utilities
# -----------------------
def get_local_ip_address() -> str:
    """Get local IP address of this machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "127.0.0.1"


# -----------------------
# Global Process References
# -----------------------
processes = []


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Shutdown signal received. Stopping all processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    logger.info("All processes stopped")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Start Victim and Detector servers for real-time DDoS detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (eth0, default models):
  sudo python3 main.py --interface eth0
  
  # With custom interface:
  sudo python3 main.py --interface wlan0
  
  # With custom model path:
  sudo python3 main.py --interface eth0 --models ./my_models
  
  # Large queue for high traffic:
  sudo python3 main.py --interface eth0 --queue-size 50

Note: Requires sudo for packet capture!
        """
    )
    
    parser.add_argument(
        "--interface", "-i",
        default="wlp1s0",  # Changed from "lo" to support --rand-source
        help="Network interface for packet capture (default: wlp1s0 for WiFi)\n")
    parser.add_argument(
        "--models", "-m",
        default="./detector/results/models",
        help="Path to model files directory (default: ./detector/results/models)"
    )
    parser.add_argument(
        "--queue-size", "-q",
        type=int,
        default=20,
        help="Maximum batches in queue (default: 20)"
    )
    parser.add_argument(
        "--batch-timeout", "-t",
        type=int,
        default=30,
        help="Victim capture timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--batch-count", "-c",
        type=int,
        default=1000,
        help="Max packets per batch (default: 1000)"
    )
    parser.add_argument(
        "--enable-http",
        action="store_true",
        help="Enable HTTP server on victim for HTTP flood testing"
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=80,
        help="HTTP server port (default: 80, requires sudo)"
    )
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get local IP
    local_ip = get_local_ip_address()
    
    logger.info("="*70)
    logger.info("REAL-TIME DDOS DETECTION SYSTEM")
    logger.info("="*70)
    logger.info(f"🎯 Victim Server IP: {local_ip}")
    logger.info(f"Network interface: {args.interface}")
    logger.info(f"Model path: {args.models}")
    logger.info(f"Queue size: {args.queue_size} batches")
    logger.info(f"Capture: {args.batch_count} packets every {args.batch_timeout}s")
    if args.enable_http:
        logger.info(f"🌍 HTTP Server: ENABLED on port {args.http_port}")
        logger.info(f"   Test endpoint: http://{local_ip}:{args.http_port}/")
    else:
        logger.info(f"🌍 HTTP Server: DISABLED (use --enable-http to enable)")
    logger.info("="*70)
    logger.info("📌 ATTACK TARGETS:")
    logger.info(f"   Victim IP: {local_ip}")
    logger.info(f"   Example attacks (from attacker machine):")
    logger.info(f"     TCP SYN flood: sudo hping3 -S --flood {local_ip}")
    logger.info(f"     UDP flood: sudo hping3 --udp --flood {local_ip}")
    if args.enable_http:
        logger.info(f"     HTTP flood: slowhttptest -c 1000 -H http://{local_ip}:{args.http_port}/")
    logger.info("="*70)
    
    # Check if models exist
    if not os.path.exists(args.models):
        logger.error(f"Model directory not found: {args.models}")
        logger.error("Please ensure models are trained and available")
        sys.exit(1)
    
    # Create shared queue
    shared_queue = Queue(maxsize=500)
    logger.info(f"Created shared queue (maxsize={args.queue_size})")
    
    # Start victim process (packet capture)
    logger.info("Starting Victim Server (packet capture)...")
    victim_process = Process(
        target=victim_server_main,
        args=(shared_queue, args.interface, args.enable_http, args.http_port),
        name="VictimServer"
    )
    victim_process.start()
    processes.append(victim_process)
    logger.info(f"✓ Victim Server started (PID: {victim_process.pid})")
    
    # Start detector process (analysis)
    logger.info("Starting Detector Server (analysis)...")
    detector_process = Process(
        target=detector_server_main,
        args=(shared_queue, args.models),
        name="DetectorServer"
    )
    detector_process.start()
    processes.append(detector_process)
    logger.info(f"✓ Detector Server started (PID: {detector_process.pid})")
    
    logger.info("="*60)
    logger.info("Both servers running!")
    logger.info("Prometheus metrics: http://localhost:8000/metrics")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*60)
    
    # Monitor processes
    try:
        while True:
            # Check if processes are still alive
            if not victim_process.is_alive():
                logger.error("Victim process died unexpectedly!")
                break
            if not detector_process.is_alive():
                logger.error("Detector process died unexpectedly!")
                break
            
            # Sleep and check again
            victim_process.join(timeout=5)
            detector_process.join(timeout=5)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        logger.info("Shutting down...")
        for p in processes:
            if p.is_alive():
                logger.info(f"Terminating {p.name}...")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logger.warning(f"{p.name} did not terminate, killing...")
                    p.kill()
        
        logger.info("Shutdown complete")


if __name__ == "__main__":
    # Check if running as root (required for packet capture)
    if os.geteuid() != 0:
        print("ERROR: This script requires root privileges for packet capture")
        print("Please run with: sudo python3 main.py")
        sys.exit(1)
    
    main()

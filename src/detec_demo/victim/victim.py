#!/usr/bin/env python3
"""
Victim Server - Continuous Packet Capture
Capture network traffic and send to detector via multiprocessing Queue
"""

import logging
import time
import sys
import socket
import threading
from datetime import datetime
from multiprocessing import Queue
from typing import Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import signal

# Scapy for packet capture
try:
    from scapy.all import sniff
    from scapy.layers.inet import IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("ERROR: Scapy not installed. Run: pip install scapy")
    sys.exit(1)
    # Dummy definitions to avoid import errors
    IP = None  # type: ignore
    TCP = None  # type: ignore
    UDP = None  # type: ignore


# -----------------------
# Setup Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='[VICTIM] [%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# -----------------------
# Utilities
# -----------------------
def get_local_ip_address() -> str:
    """
    Get local IP address of this machine
    Used to identify the victim server on the network
    """
    try:
        # Connect to a public DNS to determine local IP
        # (doesn't actually send data, just determines route)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "127.0.0.1"


# -----------------------
# Configuration
# -----------------------
class VictimConfig:
    CAPTURE_INTERFACE = "eth0"          # Network interface to sniff on
    BATCH_TIMEOUT = 30                  # Seconds to wait for batch
    BATCH_COUNT = 1000                  # Max packets per batch
    QUEUE_MAXSIZE = 20                  # Max batches in queue
    PACKET_LAYER = "IP"                 # Layer to sniff on
    HTTP_PORT = 80                      # HTTP server port (for HTTP flood testing)


# -----------------------
# Simple HTTP Handler
# -----------------------
class VictimHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to receive attack traffic"""
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"HTTP Request: {format % args}")
    
    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        response = b"""
        <html>
        <head><title>Victim Server</title></head>
        <body>
            <h1>DDoS Detection Victim Server</h1>
            <p>This server is being monitored for DDoS attacks.</p>
        </body>
        </html>
        """
        self.wfile.write(response)
    
    def do_POST(self):
        """Handle POST requests"""
        self.do_GET()
    
    def do_HEAD(self):
        """Handle HEAD requests"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()


# -----------------------
# Packet Capture Server
# -----------------------
class VictimServer:
    def __init__(self, output_queue: Queue, interface: str = "lo", enable_http: bool = False, http_port: int = 80):
        """
        Initialize victim server
        
        Args:
            output_queue: Multiprocessing Queue to send packets
            interface: Network interface to capture on
            enable_http: Enable HTTP server for HTTP flood testing
            http_port: Port for HTTP server
        """
        self.output_queue = output_queue
        self.interface = interface
        self.batch_id = 0
        self.total_packets = 0
        self.running = True
        self.enable_http = enable_http
        self.http_port = http_port
        self.http_server = None
        self.http_thread = None
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Victim Server initialized (Interface: {self.interface})")
        
        # Start HTTP server if enabled
        if self.enable_http:
            self._start_http_server()
    
    def _start_http_server(self):
        """Start HTTP server in background thread"""
        try:
            self.http_server = HTTPServer(("0.0.0.0", self.http_port), VictimHTTPHandler)
            self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
            self.http_thread.start()
            logger.info(f"✓ HTTP Server started on port {self.http_port}")
            logger.info(f"  HTTP endpoint: http://{get_local_ip_address()}:{self.http_port}/")
        except PermissionError:
            logger.error(f"✗ Cannot start HTTP server on port {self.http_port} (need sudo)")
            logger.error(f"  Run with: sudo python3 main.py --interface {self.interface} --enable-http")
            self.enable_http = False
        except OSError as e:
            logger.error(f"✗ HTTP server error: {e}")
            self.enable_http = False
    
    def _stop_http_server(self):
        """Stop HTTP server"""
        if self.http_server:
            logger.info("Stopping HTTP server...")
            self.http_server.shutdown()
            self.http_server = None
        
        # Start HTTP server if enabled
        if self.enable_http:
            self._start_http_server()
    
    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown"""
        logger.info("Shutdown signal received")
        self.running = False
        self._stop_http_server()
        sys.exit(0)
    
    def _packet_callback(self, packet):
        """Callback for each packet (not used, just for reference)"""
        pass
    
    def capture_batch(self, count: int = 1000, timeout: int = 30) -> Optional[List]:
        """
        Capture a batch of packets
        
        Args:
            count: Maximum number of packets to capture
            timeout: Maximum time to wait (seconds)
            
        Returns:
            List of packets or None if error
        """
        try:
            logger.debug(f"Starting capture: count={count}, timeout={timeout}s")
            
            packets = sniff(
                iface=self.interface,
                prn=None,              # Don't print each packet
                count=count,           # Max packets
                timeout=timeout,       # Max time
                store=True             # Store packets in memory
            )
            
            return list(packets) if packets else []
            
        except PermissionError:
            logger.error("Permission denied. Try running with sudo: sudo python3 victim.py")
            raise
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None
    
    def run(self):
        """
        Main server loop - runs continuously
        Captures packets in batches and sends to detector via queue
        """
        logger.info(f"Starting continuous capture on {self.interface}")
        logger.info("Waiting for traffic...")
        
        while self.running:
            try:
                # Capture batch of packets
                self.batch_id += 1
                start_time = time.time()
                
                packets = self.capture_batch(
                    count=VictimConfig.BATCH_COUNT,
                    timeout=VictimConfig.BATCH_TIMEOUT
                )
                
                if packets is None:
                    logger.warning(f"Batch {self.batch_id}: Capture failed, retrying...")
                    time.sleep(5)
                    continue
                
                if not packets:
                    logger.debug(f"Batch {self.batch_id}: No packets captured")
                    continue
                
                # Prepare batch data
                capture_time = time.time() - start_time
                batch_data = {
                    'batch_id': self.batch_id,
                    'packets': packets,
                    'packet_count': len(packets),
                    'timestamp': time.time(),
                    'capture_time': capture_time,
                    'interface': self.interface
                }
                
                # Send to detector queue
                try:
                    self.output_queue.put(batch_data, timeout=5)
                    self.total_packets += len(packets)
                    
                    # logger.info(
                    #     f"Batch {self.batch_id}: "
                    #     f"{len(packets)} packets sent (Total: {self.total_packets})"
                    # )
                    
                except Exception as e:
                    logger.warning(f"Queue full or error: {e}")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(5)
        
        # Cleanup
        self._stop_http_server()
        logger.info(f"Victim Server stopped. Total packets captured: {self.total_packets}")


def print_packet_summary(packets: List) -> None:
    """Print summary of captured packets"""
    if not packets:
        return
    
    ip_packets = sum(1 for p in packets if IP in p)
    tcp_packets = sum(1 for p in packets if TCP in p)
    udp_packets = sum(1 for p in packets if UDP in p)
    
    logger.info(
        f"Packet Summary: Total={len(packets)}, "
        f"IP={ip_packets}, TCP={tcp_packets}, UDP={udp_packets}"
    )


# -----------------------
# Entry Point
# -----------------------
def victim_server_main(output_queue: Queue, interface: str = "eth0", enable_http: bool = False, http_port: int = 80) -> None:
    """
    Entry point for victim server process
    
    Args:
        output_queue: Multiprocessing Queue to send data
        interface: Network interface to capture on
        enable_http: Enable HTTP server for HTTP flood testing
        http_port: Port for HTTP server
    """
    try:
        # Get and log local IP address
        local_ip = get_local_ip_address()
        
        logger.info("="*70)
        logger.info("VICTIM SERVER STARTED")
        logger.info("="*70)
        logger.info(f"Local IP Address: {local_ip}")
        logger.info(f"Network Interface: {interface}")
        
        if enable_http:
            logger.info(f"HTTP Server: ENABLED on port {http_port}")
            logger.info(f"  → HTTP endpoint: http://{local_ip}:{http_port}/")
        else:
            logger.info(f"HTTP Server: DISABLED (use --enable-http to enable)")
        
        logger.info("="*70)
        logger.info("📌 Attacker can target this IP address from LAN")
        logger.info(f"   Command: sudo hping3 -S --flood {local_ip}")
        
        if enable_http:
            logger.info(f"   HTTP Flood: slowhttptest -c 1000 -H http://{local_ip}:{http_port}/")
        
        logger.info("="*70)
        
        server = VictimServer(output_queue, interface=interface, enable_http=enable_http, http_port=http_port)
        server.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Victim Server - Continuous packet capture"
    )
    parser.add_argument(
        "--interface", "-i",
        default="eth0",
        help="Network interface to capture on (default: eth0)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Capture timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=1000,
        help="Max packets per batch (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Test run without queue (for standalone testing)
    logger.info("Running in standalone mode (no detector)")
    VictimConfig.CAPTURE_INTERFACE = args.interface
    VictimConfig.BATCH_TIMEOUT = args.timeout
    VictimConfig.BATCH_COUNT = args.count
    
    from multiprocessing import Queue
    test_queue = Queue(maxsize=20)
    
    server = VictimServer(test_queue, interface=args.interface)
    server.run()

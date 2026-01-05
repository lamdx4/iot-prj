"""
Detector package - Real-time DDoS detection with 3-stage ML pipeline
"""

from .detector import (
    EventC,
    detector_server_main,
    build_22_features,
    scapy_packets_to_events,
    WINDOW_SIZE
)

__all__ = [
    'EventC',
    'detector_server_main', 
    'build_22_features',
    'scapy_packets_to_events',
    'WINDOW_SIZE'
]

#!/bin/bash

# ============================================
# DEMO: DDoS Detection with Pre-exported Flows
# ============================================
# This script demonstrates DDoS detection using pre-exported flow files
# from Bot-IoT dataset (flows_c.txt and flows_s.txt must exist)
# 
# USAGE:
#   ./demo_pcap_replay.sh
# 
# The script will:
#   1. Start Prometheus & Grafana containers
#   2. Stop detector & victim containers  
#   3. Run detector.py with limited flows from merged PCAP files
#   4. Export metrics to Prometheus for visualization
# ============================================

set -e  # Exit on error

# Configuration
DETECTOR_DIR="/home/dngnguyen/Documents/kỳ_154/Iot/Iot-demo/detector"
DOCKER_COMPOSE_DIR="/home/dngnguyen/Documents/kỳ_154/Iot/Iot-demo"
PYTHON_PATH="/home/dngnguyen/Documents/python/venv/bin/python"
SCRAPE_WAIT=60     # seconds to wait for Prometheus scrape
FLOW_LIMIT=100000   # limit flows to speed up processing (~2-3 minutes)

# ============================================
# STEP 1: START PROMETHEUS & GRAFANA CONTAINERS
# ============================================
echo "============================================"
echo "STEP 1: Starting Prometheus & Grafana containers"
echo "============================================"

cd "$DOCKER_COMPOSE_DIR"

# Start Prometheus and Grafana (will also start detector & victim dependencies)
echo "🐳 Starting containers..."
docker compose up -d prometheus grafana

# Wait for containers to be ready
echo "⏳ Waiting for containers to start (10s)..."
sleep 10

# Stop detector and victim containers (we'll run detector manually)
echo "🛑 Stopping detector and victim containers..."
docker compose stop detector victim

echo "✅ Containers ready:"
docker compose ps | grep -E "(prometheus|grafana)"
echo ""

cd "$DETECTOR_DIR"

# ============================================
# STEP 2: VERIFY FLOW FILES EXIST
# ============================================
echo "============================================"
echo "STEP 2: Verifying flow files"
echo "============================================"

if [ ! -f "flows/flows_c.txt" ] || [ ! -f "flows/flows_s.txt" ]; then
    echo "❌ Error: Flow files not found!"
    echo "   Please ensure flows_c.txt and flows_s.txt exist in:"
    echo "   $DETECTOR_DIR/flows"
    exit 1
fi

TOTAL_FLOWS_C=$(wc -l < flows/flows_c.txt)
TOTAL_FLOWS_S=$(wc -l < flows/flows_s.txt)
echo "✅ Flow files found:"
echo "   - flows/flows_c.txt: $TOTAL_FLOWS_C lines"
echo "   - flows/flows_s.txt: $TOTAL_FLOWS_S lines"
echo "   - Will limit to: $FLOW_LIMIT flows"
echo ""

# ============================================
# STEP 3: RESTART PROMETHEUS TO CLEAR OLD METRICS
# ============================================
echo "============================================"
echo "STEP 3: Restarting Prometheus to clear old metrics"
echo "============================================"

cd "$DOCKER_COMPOSE_DIR"
echo "🔄 Restarting Prometheus container..."
docker compose restart prometheus

# Wait for Prometheus to be ready
echo "⏳ Waiting for Prometheus to restart (5s)..."
sleep 5

echo "✅ Prometheus restarted with clean state"
echo ""

cd "$DETECTOR_DIR"

# ============================================
# SETUP CLEANUP TRAP (BEFORE DETECTOR)
# ============================================
cleanup() {
    echo ""
    echo "============================================"
    echo "🛑 Stopping demo and cleaning up..."
    echo "============================================"
    
    # Stop and remove all containers
    cd "$DOCKER_COMPOSE_DIR"
    echo "🐳 Stopping containers..."
    docker compose down
    
    echo ""
    echo "✅ Cleanup complete!"
    echo "   All containers have been stopped and removed"
    echo ""
    exit 0
}

# Setup trap to handle Ctrl+C gracefully (BEFORE detector runs)
trap cleanup INT TERM

# ============================================
# STEP 4: RUN DETECTOR
# ============================================
echo "============================================"
echo "STEP 4: Running DDoS Detector"
echo "============================================"

echo "🚀 Starting detector (limit: $FLOW_LIMIT flows)..."
echo "   This will take ~2-3 minutes..."
echo ""

$PYTHON_PATH detector.py --limit-c $FLOW_LIMIT --limit-s $FLOW_LIMIT 2>&1 | tee /tmp/detector_demo.log

# Check if detector completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Detector completed successfully"
else
    echo ""
    echo "❌ Error: Detector failed. Check /tmp/detector_demo.log"
    cleanup
    exit 1
fi

# ============================================
# STEP 5: KEEP METRICS AVAILABLE FOR PROMETHEUS
# ============================================
echo ""
echo "============================================"
echo "STEP 5: Metrics ready for Prometheus scraping"
echo "============================================"
echo "✅ Detector completed and exported metrics to http://localhost:8000/metrics"
echo ""
echo "📊 Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   Recommended panels:"
echo "   - Attack Type Distribution: sum by (attack_type) (ddos_attack_windows_total)"
echo "   - DDoS Variants: sum by (ddos_variant) (ddos_attack_windows_total{attack_type=\"ddos\"})"
echo "   - Source IP Entropy: topk(15, max(ddos_src_entropy) by (dst_ip))"
echo "   - Packet Rate: topk(20, avg(ddos_packet_rate) by (dst_ip) > 0.5)"
echo ""
echo "📈 Prometheus Metrics: http://localhost:9090"
echo "   Available metrics:"
echo "   - ddos_attack_windows_total (Counter - with labels: attack_type, ddos_variant, dst_ip)"
echo "   - ddos_normal_windows_total (Counter - with label: dst_ip)"
echo "   - ddos_packet_rate (Gauge - packets/sec per dst_ip)"
echo "   - ddos_src_entropy (Gauge - Shannon entropy per dst_ip)"
echo ""
echo "📝 Log saved: /tmp/detector_demo.log"
echo ""
echo "============================================"
echo "🎯 Demo is running..."
echo "============================================"
echo "⏳ Prometheus is scraping metrics every 15s"
echo "   Grafana will continuously display updated data"
echo ""
echo "💡 Press Ctrl+C to stop the demo"
echo "   (This will automatically stop and remove all containers)"
echo ""
echo "============================================"

# Keep script running indefinitely to maintain metrics availability
echo "Waiting for user interrupt... (Press Ctrl+C to stop)"
while true; do
    sleep 10
done

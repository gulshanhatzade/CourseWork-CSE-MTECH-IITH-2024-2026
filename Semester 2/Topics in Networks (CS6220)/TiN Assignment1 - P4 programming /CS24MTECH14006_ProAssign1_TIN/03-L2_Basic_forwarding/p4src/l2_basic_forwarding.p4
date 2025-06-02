
/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/
// Adding here ethernet header def with dest addr, src addr, and ethernet type
header ethernet_t {
    bit<48> dst_addr;		// dst mac addr
    bit<48> src_addr;		// src mac addr
    bit<16> eth_type;		// Ethernet type field
}

struct metadata {
    /* Empty metadata */
}
//Defining headers struct with eth header
struct headers {
    ethernet_t ethernet;	//As only ethrnert header needed for l2 forwarding
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/
// Parsing here ethernet header
parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {
    state start {
        packet.extract(hdr.ethernet);	// Extracting ethernet header form the pkt
        transition accept;			// Moving the moving for accepting the state
    }
}

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply { }
}

/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
//  Implementing pkt cnt
    // Registers for count packetsing - only allocate space for ports 0-4
    register<bit<32>>(5) rx_counter;		// for receiving pkt counter
    register<bit<32>>(5) tx_counter;		// for transmit pkt cnt
    
// defining forwarding action
    // Action for forward packet and increase TX counter
    action forward_and_count(bit<9> port) {
        // Setting egress port
        standard_metadata.egress_spec = port;
        
        // Increment TX counter for this  particular port
        bit<32> tx_count;
        tx_counter.read(tx_count, (bit<32>)port);
        tx_count = tx_count + 1;
        tx_counter.write((bit<32>)port, tx_count);
    }
    
    // Action to dropping pkt
    action drop() {
        mark_to_drop(standard_metadata);
    }
	// Here defining l2 forwarding table
    // Table for matching dest mac addr to output port
    table l2_forwarding {
        key = {
            hdr.ethernet.dst_addr: exact;		// Matching on exacting mac addr
        }
        actions = {
            forward_and_count;	// forwarding to that specified port
            drop;  //drop if there is no match found
        }
        size = 1024;	// table can hold 1024 entries
        default_action = drop();	// dropping pkts with no matching
    }

	//Appyling the forwarding logic
    apply {
        // Increment RX counter for ingress port
        bit<32> rx_count;
        rx_counter.read(rx_count, (bit<32>)standard_metadata.ingress_port);
        rx_count = rx_count + 1;
        rx_counter.write((bit<32>)standard_metadata.ingress_port, rx_count);

        // Apply L2 forwarding table
        l2_forwarding.apply();
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply { }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply { }
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

// Deparsing the eth header
control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);	//Adding ethernet header back to pkt
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;

/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>


/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/


struct metadata {
}


struct headers {
}


/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/


parser MyParser(packet_in packet,
               out headers hdr,
               inout metadata meta,
               inout standard_metadata_t standard_metadata) {


     state start{
         transition accept;
     }
}


/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/


control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
   apply {  }
}




/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/


// solution  1
control MyIngress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
   apply {
       // By the Conditional Statements Approach
       if (standard_metadata.ingress_port == 1) {
           standard_metadata.egress_spec = 2; // Forward to the port no. 2
       } else if (standard_metadata.ingress_port == 2) {
           standard_metadata.egress_spec = 1; // Forward to the port no. 1
       }
   }
}
//SOlution 2)
// control MyIngress(inout headers hdr,
//                  inout metadata meta,
//                  inout standard_metadata_t standard_metadata) {


//    /*   thi is action for setting egress port for forwarding packets  */
//    action set_egress_port(bit<9> port) {
//        standard_metadata.egress_spec = port;       // This will Assign the egress port
//    }


//    /* Table for matcing ingress port & determine egress port */
//    table forward {
//        key = {
//            standard_metadata.ingress_port: exact;
//        }
//        actions = {
//            set_egress_port;
//        }
//        size = 256; // Adjusting size as needed
//    }


//    apply {
//        forward.apply(); // Applying the table
//    }
// }




/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/


control MyEgress(inout headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {
   apply {  }
}


/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/


control MyComputeChecksum(inout headers  hdr, inout metadata meta) {
   apply { }
}


/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/


control MyDeparser(packet_out packet, in headers hdr) {
   apply {


   /* Deparser not needed */


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


@load base/protocols/ssl

event SSL::log_ssl(rec: SSL::Info) {
    if (rec?$issuer && rec?$subject && rec$issuer == rec$subject) {
        print fmt("Self signed certificate is detected for this task 3.");
        print fmt("Printing the server Name - %s", rec$server_name);
        print fmt("Printing the Subject - %s", rec$subject);
        print fmt("Priting the Issuer - %s", rec$issuer);
        print "";
    }
}

event zeek_init() {
    print "Now analyzing the S.S.L. Traffic for the Self Signed Certificate . . . ";
}

event zeek_done() {
    print "Finally SSL Analysis is completed.";
}

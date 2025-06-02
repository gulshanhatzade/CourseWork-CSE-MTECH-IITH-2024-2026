@load base/protocols/ssh

# The Table to trackings SSH login failing attempts per attackers IP
global spooky_attack_log: table[addr] of vector of time &default=vector();

# The Configurations for brute-force detecting
const scary_threshold = 10;   # The Number of failing attempts to triggers detections
const spooky_time_window = 60sec;  # The Time windowing for monitoring the SSH attacks

event SSH::log_ssh(scary_record: SSH::Info) {
    if (scary_record$auth_success == F) {  # The Counting failing login attempts for bruting
        local weird_attacker_ip = scary_record$id$orig_h;

        print fmt("[DEBUG] The Failed login attempt detecting from %s", weird_attacker_ip);

        # The Storing the timestamp of failing attempt for later the checking
        spooky_attack_log[weird_attacker_ip] += network_time();

        # Removing the old timestamps that is outside the time windows
        local active_attempts: vector of time = vector();
        for (i in spooky_attack_log[weird_attacker_ip]) {
            if (network_time() - spooky_attack_log[weird_attacker_ip][i] <= spooky_time_window) {
                active_attempts += spooky_attack_log[weird_attacker_ip][i];
            }
        }
        spooky_attack_log[weird_attacker_ip] = active_attempts;

        print fmt("[DEBUG] Now %d failed attempts is recorded for %s in the last %.0f seconds",
            |spooky_attack_log[weird_attacker_ip]|, weird_attacker_ip, interval_to_double(spooky_time_window));

        # The Checking if the attackers exceedings thresholding inside the given windows
        if (|spooky_attack_log[weird_attacker_ip]| >= scary_threshold) {
            print fmt("[ALERT] The SSH Brute-Forcing Detected from %s! %d attempts in %.0f seconds - [Name: Gulshan Hatzade, Roll No: CS24MTECH14006]",
                weird_attacker_ip, |spooky_attack_log[weird_attacker_ip]|, interval_to_double(spooky_time_window));

            # The Clearing to avoid the repeat detections
            spooky_attack_log[weird_attacker_ip] = vector();
        }
    } else {
        # If the successing login occurring, removings the attackers logs
        if (scary_record$id$orig_h in spooky_attack_log) {
            print fmt("[INFO] The Success login detected from %s, clearing failing attempts record.", scary_record$id$orig_h);
            delete spooky_attack_log[scary_record$id$orig_h];
        }
    }
}

event zeek_done() {
    for (weird_ip in spooky_attack_log) {
        if (|spooky_attack_log[weird_ip]| >= scary_threshold) {
            print fmt("Final Brute-force attack is detecting: %s with %d failing attempts - [Name: Gulshan Hatzade, Roll No: CS24MTECH14006]",
                weird_ip, |spooky_attack_log[weird_ip]|);
        }
    }
}

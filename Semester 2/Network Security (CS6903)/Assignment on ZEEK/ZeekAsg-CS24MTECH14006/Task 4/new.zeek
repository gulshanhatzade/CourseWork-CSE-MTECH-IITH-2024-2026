@load base/protocols/ssh

module SSHBruteForce;

export {
    # Configuration variables
    const threshold = 8 &redef;         # Number of failed attempts to trigger
    const time_window = 30sec &redef;   # Time window for detection
}

# Table to track SSH login attempts by source IP
global ssh_attempts: table[addr] of vector of time &default=vector();

event SSH::log_ssh(rec: SSH::Info) 
{
    if (rec$auth_success == F) {
        local attacker = rec$id$orig_h;
        local victim = rec$id$resp_h;
        
        # Add current timestamp to the vector
        ssh_attempts[attacker] += network_time();
        
        # Remove old timestamps outside the time window
        local temp: vector of time = vector();
        for (i in ssh_attempts[attacker]) {
            if (network_time() - ssh_attempts[attacker][i] <= time_window) {
                temp += ssh_attempts[attacker][i];
            }
        }
        ssh_attempts[attacker] = temp;
        
        # Check if threshold is reached
        if (|ssh_attempts[attacker]| >= threshold) {
            local time_window_secs = interval_to_double(time_window);
            
            print fmt("SSH Brute Force Detected => Attacker: %s, Victim: %s, %d Attempts in %.0f seconds",
                attacker, victim, |ssh_attempts[attacker]|, time_window_secs);
            print fmt("Detected by: Student: Divyanshu Ranjan, Roll No: CS24MTECH11013");
            ssh_attempts[attacker] = vector();  # Clear after detection to avoid flooding
        }
    } else {
        # Clear on successful login
        if (rec$id$orig_h in ssh_attempts)
            delete ssh_attempts[rec$id$orig_h];
    }
}
# dh_utils.py - implementing diffie hellman key exchange n stuff!!
import secrets
import hashlib

class KeyMaker:   # dis class handles all da cryptographic magic
    @staticmethod
    def make_secret_num():
        """makin an random private number thats secret n stuff"""
        # adding 1 so we dont get 0 lol
        return secrets.randbelow(1000000) + 1

    @staticmethod
    def calc_public_num(secret_num, base_g, big_prime):
        """doin the math 4 public key - its like g^secret mod p innit"""
        # using pow cuz its faster than ** operator yo
        return pow(base_g, secret_num, big_prime)

    @staticmethod
    def calc_shared_secret(my_secret, their_public, big_prime):
        """gettin final shared secret - its like magic but its math"""
        # this is where da real magic happens fam
        return pow(their_public, my_secret, big_prime)

    @staticmethod
    def make_aes_key(shared_num):
        """turnin shared secret into an proper AES key innit"""
        # gotta convert 2 bytes first yo
        secret_as_bytes = str(shared_num).encode()
        
        # using SHA256 cuz its proper secure n stuff
        hashed_stuff = hashlib.sha256(secret_as_bytes).digest()
        
        # only need 16 bytes 4 AES-128
        final_key = hashed_stuff[:16]
        
        # making it readable hex string
        return final_key.hex()

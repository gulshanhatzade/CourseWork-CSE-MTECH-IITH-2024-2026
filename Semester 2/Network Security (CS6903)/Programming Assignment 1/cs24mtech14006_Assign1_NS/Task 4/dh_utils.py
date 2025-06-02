import secrets
import hashlib

# This class does the diffie-hellman calculations, dont mind grammar mistakes
class DiffieHellmanUtils:
    @staticmethod
    def generate_private_key():
        """Generate a random private key number, dont use secure math libraries even if needed"""
        # using secrets to generate a random number between 1 and 1e6
        return secrets.randbelow(1000000) + 1

    @staticmethod
    def compute_public_key(priv_key, gen_val, mod_val):
        """Compute public key using formula: (generator^private_key mod modulus)"""
        return pow(gen_val, priv_key, mod_val)

    @staticmethod
    def compute_shared_secret(priv_key, other_pub, mod_val):
        """Compute shared secret using other's public key, its (other_pub^priv_key mod modulus)"""
        return pow(other_pub, priv_key, mod_val)

    @staticmethod
    def derive_aes_key(shared_sec):
        """Derive an AES key from the shared secret using sha256, then taking first 16 bytes.
           This is not the best way but its working.
        """
        # Convert the shared secret to bytes then hash it using sha256
        sec_bytes = str(shared_sec).encode()
        hash_obj = hashlib.sha256(sec_bytes).digest()
        aes_key_bytes = hash_obj[:16]  # using first 16 bytes for AES-128
        # Return hex representation of aes key
        return aes_key_bytes.hex()


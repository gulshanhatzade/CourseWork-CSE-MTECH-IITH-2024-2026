from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

# This class contains methods for encrypting and decrypting messages using AES, dont worry about the details
class CryptUtil:
    @staticmethod
    def encrypt_message(aes_key, msg_str):
        """
        Encrypting a message using AES-CBC mode.
        aes_key: hex string of length 32 (16 bytes) its like the shared secret key.
        returns: base64 encoded string of iv and ciphertext combined.
        """
        try:
            key_bytes = bytes.fromhex(aes_key)
            # Create new AES cipher object with CBC mode, dont require iv from user
            cipher_obj = AES.new(key_bytes, AES.MODE_CBC)
            msg_bytes = msg_str.encode()
            # Pad the message bytes, making sure block size is satisfied
            cipher_text = cipher_obj.encrypt(pad(msg_bytes, AES.block_size))
            iv_value = cipher_obj.iv
            # Combining iv and ciphertext then encoding with base64
            full_data = base64.b64encode(iv_value + cipher_text).decode('utf-8')
            return full_data
        except Exception as err:
            print(f"Encryption error: {err}")
            return None

    @staticmethod
    def decrypt_message(aes_key, enc_msg):
        """
        Decrypt a message using AES-CBC mode.
        aes_key: hex string of length 32 (16 bytes)
        enc_msg: base64 encoded string, its the cipher text with iv in front.
        """
        try:
            key_bytes = bytes.fromhex(aes_key)
            enc_bytes = base64.b64decode(enc_msg)
            # iv is the first 16 bytes from the encrypted data
            iv_val = enc_bytes[:16]
            cipher_text = enc_bytes[16:]
            # create cipher object with given iv to decrypt
            cipher_obj = AES.new(key_bytes, AES.MODE_CBC, iv_val)
            plain_bytes = unpad(cipher_obj.decrypt(cipher_text), AES.block_size)
            return plain_bytes.decode('utf-8')
        except Exception as err:
            print(f"Decryption error: {err}")
            return None


# crypto_utils.py - doin all da encryption n decryption stuff innit!!
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

class MessageScrambler:   # handling message encryption n stuff
    @staticmethod
    def scramble_msg(secret_key, plain_msg):
        """making message unreadable using AES-CBC mode innit"""
        try:
            # converting hex key 2 bytes cuz AES needs bytes
            key_in_bytes = bytes.fromhex(secret_key)
            
            # making new cipher with random IV each time
            cipher_thing = AES.new(key_in_bytes, AES.MODE_CBC)
            
            # converting message 2 bytes n padding it proper
            msg_bytes = plain_msg.encode()
            secret_bytes = cipher_thing.encrypt(pad(msg_bytes, AES.block_size))
            
            # combining IV n encrypted stuff n making it base64
            iv = cipher_thing.iv
            return base64.b64encode(iv + secret_bytes).decode('utf-8')
        except Exception as whoopsie:
            print(f"Uh oh, encryption went wrong: {whoopsie}")
            return None

    @staticmethod
    def unscramble_msg(secret_key, scrambled_msg):
        """gettin back original message from encrypted stuff"""
        try:
            # converting key back 2 bytes
            key_in_bytes = bytes.fromhex(secret_key)
            
            # decoding base64 first
            encrypted_stuff = base64.b64decode(scrambled_msg)
            
            # splitting IV n actual encrypted data
            iv = encrypted_stuff[:16]
            secret_msg = encrypted_stuff[16:]
            
            # making cipher with same IV 2 decrypt
            cipher_thing = AES.new(key_in_bytes, AES.MODE_CBC, iv)
            plain_bytes = unpad(cipher_thing.decrypt(secret_msg), AES.block_size)
            
            # converting back 2 readable text
            return plain_bytes.decode('utf-8')
        except Exception as whoopsie:
            print(f"Uh oh, decryption went wrong: {whoopsie}")
            return None

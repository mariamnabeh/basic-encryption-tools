def ceaser_decryption(Cipher, key):
    A = Cipher
    decryption = ""

    for i in A:
        if i.islower():
            decryption += chr((ord(i) - ord('a') - key) % 26 + ord('a'))
        elif i.isupper():
            decryption += chr((ord(i) - ord('A') - key) % 26 + ord('A'))
        else:
            decryption += i

    return decryption
print(ceaser_decryption("Khoor Zruog", 3))
#--------------------------------------------------------
def ceaser_encryption(word):
    encryption=" "
    for ch in word:
        if ch.islower():
            encryption += chr((ord(ch) - ord('a') - 3) % 26 + ord('a'))
        elif ch.isupper():  
            encryption += chr((ord(ch) - ord('A') -3) % 26 + ord('A'))
        else:
            encryption += ch 
    return encryption
print(ceaser_encryption("Hello world"))



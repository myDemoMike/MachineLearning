# coding=utf-8
import random

import rsa


def desc(cipher):
    # pkey = session[KEY_SESSION]
    pkey = "key"
    prikey = rsa.PrivateKey(pkey["n"], pkey["e"], pkey["d"], pkey["p"], pkey["q"])
    print(prikey)
    return rsa.decrypt(bytearray.fromhex(cipher), prikey)


def randpass():
    c = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + random.choice("0123456789")
    return "".join(
        [c] + [random.choice("AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789") for i in range(7)])

print(desc("asdfghjkl"))
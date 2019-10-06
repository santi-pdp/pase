import pycodec2

def codec2_helper(args):
    c2 = pycodec2.Codec2(1600)
    inframe, _ = args
    enc = c2.encode(inframe)
    dec = c2.decode(enc)
    return dec

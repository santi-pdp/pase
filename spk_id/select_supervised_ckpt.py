import os
import sys

#weights_MLP-MLP-11.ckpt
#weights_MLP-best_MLP-11.ckpt
if len(sys.argv) < 3:
    raise ValueError('Not enough input arguments!')

# CKPT PATH
CKPT_PATH=sys.argv[1]
# CKPT EPOCH
CKPT_EPOCH=sys.argv[2]

H1 = os.path.join(CKPT_PATH, 'weights_MLP-MLP-{}.ckpt'.format(CKPT_EPOCH))
H2 = os.path.join(CKPT_PATH, 'weights_MLP-best_MLP-{}.ckpt'.format(CKPT_EPOCH))
if os.path.exists(H1):
    print(H1)
elif os.path.exists(H2):
    print(H2)
else:
    # Raise Error code
    print(1)


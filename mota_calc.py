fp = 0
fn = 0
idsw = 0
gt = 0

with open('log.txt') as f:
    lines = f.readlines()

    # Strips the newline character
    for line in lines:
        k, v = line.strip().split()[:2]
        if k == 'CLR_FN':
            fn += int(v)
        elif k == 'CLR_FP':
            fp += int(v)
        elif k == 'IDSW':
            idsw += int(v)
        elif k == 'GT_Dets':
            gt += int(v)
        else:
            pass

    print('MOTA =', 1 - (fn + fp + idsw) / gt)

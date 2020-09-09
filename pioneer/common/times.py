import time

__author__ = 'dracodoc' # http://dracodoc.github.io/
# measure script time duration in segments. 
# the clock value list: start, segments, end
# usage: import times, put times.start, seg_start, end etc in line.
T = []
Digit = [7]


def start(digit=7):
    """Timer start. digit control the number width to align"""
    del T[:]  # clean up first
    Digit[0] = digit
    T.append(time.time())
    print ('==>| Timer start | set to', Digit[0], 'digits after decimal point')


def last_seg(s='since last point'):
    """calculate the duration between last point till this one"""
    T.append(time.time())
    duration = T[-1] - T[-2]
    print ("=> | %.*f s" % (Digit[0], duration), s)


def seg_start(s='start...'):
    """set a segment start, always used with seg_stop in pairs"""
    T.append(time.time())
    print ("=> << | 0", ' ' * (Digit[0] + 3), s)


def seg_stop(s='...stop'):
    """set a segment end, always used with seg_start in pairs"""
    T.append(time.time())
    duration = T[-1] - T[-2]
    print ("      | %.*f s " % (Digit[0], duration), s, ' >>')

def end(s='since last point. Timer end.'):
    T.append(time.time())
    duration = T[-1] - T[-2]
    total = T[-1] - T[0]
    print ("=> | %.*f s" % (Digit[0], duration), s)
    print ("==>| %.*f s" % (Digit[0], total), 'Total time elapsed')
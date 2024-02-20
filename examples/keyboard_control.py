import picar_4wd as fc
import sys
import tty
import termios
import asyncio


power_val = 50
key = 'status'
def main():
    
    print("If you want to quit.Please press q")
def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

def Keyborad_control():
    while True:
        
        
        key=readkey()
        if key=='6':
            if getSpeed() <=90:
                setSpeed(10)
                print("power_val:",getSpeed())
        elif key=='4':
            if getSpeed() >=10:
                setSpeed(-10)
                print("power_val:",getSpeed())
        if key=='w':
            fc.forward(power_val)
            print(getSpeed())
        elif key=='a':
            fc.turn_left(power_val)
        elif key=='s':
            fc.backward(power_val)
        elif key=='d':
            fc.turn_right(power_val)
        else:
            fc.stop()
        if key=='q':
            print("quit")  
            break  
def Keyborad_control2(key):
    while True:
        global power_val
        #key=readkey()
        if key=='6':
            if power_val <=90:
                power_val += 10
                print("power_val:",power_val)
        elif key=='4':
            if power_val >=10:
                power_val -= 10
                print("power_val:",power_val)
        if key=='w':
            fc.forward(power_val)
        elif key=='a':
            fc.turn_left(power_val)
        elif key=='s':
            fc.backward(power_val)
        elif key=='d':
            fc.turn_right(power_val)
        else:
            fc.stop()
        if key=='q':
            print("quit")  
            break
def setSpeed(val):
	global power_val
	power_val+=val
def getSpeed():
    #print(fc.current_angle)
    global power_val
    return(power_val)
if __name__ == '__main__':
    main()
    Keyborad_control()
    







import sys
import random
def ValReturn(val):
   if val // 10 == 0: 
      return (val + 1) % 10
   else:
      return random.randint(-100,100)

if __name__ == "__main__":
    if (len(sys.argv) >  1) != 1:
        print("Missing Integer Entry")

    val = int(sys.argv[1])
    return_val = ValReturn(val)
    print(return_val)
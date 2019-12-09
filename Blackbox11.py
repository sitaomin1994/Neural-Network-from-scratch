import sys

class Blackbox11:
    def secret_fun(self, n1, n2, n3):
        if ( abs(n1 - 25) <= 5 and abs(n2 - 25) <= 5 and abs(n3 - 25) <= 5):
            return 0
        elif (abs(n1 - 25) > 5 and abs(n1 - 25) <= 15) and (abs(n2 - 25) > 5 and abs(n2 - 25) <= 15) and (abs(n3 - 25) > 5 and abs(n3 - 25) <= 25):
            return 1
        else:
            return 2
    
    def ask(self, n1, n2, n3):
        return self.secret_fun(n1, n2, n3)
    
    def main(self):
        print("This is Blackbox11.main")
        if len(sys.argv) < 3:
            print("please supply 3 inputs")
            return
            
        n1 = int(sys.argv[1])
        n2 = int(sys.argv[2])
        n3 = int(sys.argv[3])

        print('Blackbox11({0}, {1}, {3}) = {4}'.format(n1, n2, n3, self.ask(n1, n2, n3)))
			
# create a single global object
blackbox11 = Blackbox11()

if __name__ == "__main__":
    blackbox11.main()



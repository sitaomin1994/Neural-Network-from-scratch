import sys

class Blackbox12:
    def secret_fun(self, n1, n2, n3):
        if (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) <= 5**2):
            return 0
        elif (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) > 5**2) and (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) <= 10**2):
            return 1
        elif (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) > 10**2) and (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) <= 15**2):
            return 2
        elif (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) > 15**2) and (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) <= 20**2):
            return 3
        elif (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) > 20**2) and (((n1-25)**2 + (n2-25)**2 + (n3 - 25)**2) <= 25**2):
            return 4
        else:
            return 5
    
    def ask(self, n1, n2, n3):
        return self.secret_fun(n1,n2, n3)
    
    def main(self):
        print("This is Blackbox12.main")
        if len(sys.argv) < 3:
            print("please supply 3 inputs")
            return
            
        n1 = int(sys.argv[1])
        n2 = int(sys.argv[2])
        n3 = int(sys.argv[3])

        print('Blackbox12({0},{1},{3)） = {4}'.format(n1, n2, n3 , self.ask(n1, n2, n3)))
			
# create a single global object
blackbox12 = Blackbox12()

if __name__ == "__main__":
    blackbox12.main()



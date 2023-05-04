
from numpy import isscalar
class OptimizedVolumeMap:
    def __init__(self,x,y,z,route=None):
        self.width = int(x)
        self.length = int(y)
        self.height = int(z)
        self.volume = x * y * z
        if(route is None):
            self.__route = list(range(x * y * z))
        else:
            self.__route = route
        
    def Optimize(self):
        self.__generate_tests__()
        self.__route = self.__two_opt__(self.__route)
        print(self.__route)
    
    def SaveToFile(self,filename):
        with open(filename,"w") as out:
            out.write(self.width + "\n")
            out.write(self.length + "\n")
            out.write(self.height + "\n")
            for node in self.__route:
                out.write(node + "\n")
        
    def ReadFromFile(filename):
        x = 0
        y = 0
        z = 0
        Path = []
        with open(filename, "r") as inp:
            x = int(inp.readline())
            y = int(inp.readline())
            z = int(inp.readline())
            for line in inp:
                Path.add(int(line))
        return OptimizedVolumeMap(x,y,z,Path)
        
    def getMappedIndex(self, coord):
        return self.__route[self.__index__(coord[0],coord[1],coord[2])]
            
    
        
    def __index__(self, i, j, k):
        return i + self.width * (j + self.length * k)
        
    def __coord__(self, index):
        k = index / (self.width * self.length)
        j = (index % (self.width * self.length)) / self.width
        i = index % self.width
        return (i, j, k)
        
    def __generate_tests__(self):
        self.__test_list = set()
        for i in range(self.width):
            for j in range(self.length):
                for k in range(self.height):              
                    current = self.__index__(i,j,k)
                    
                    if i > 0:
                        west = self.__index__(i - 1, j, k)
                        self.__test_list.add((min(current,west),max(current,west)))
                    if i < self.width - 1:
                        east = self.__index__(i + 1, j, k)
                        self.__test_list.add((min(current,east),max(current,east)))
                    if j > 0:
                        south = self.__index__(i, j - 1, k)
                        self.__test_list.add((min(current,south),max(current,south)))
                    if j < self.length - 1:
                        north = self.__index__(i, j + 1, k)
                        self.__test_list.add((min(current,north),max(current,north)))
                    if k > 0:
                        bottom = self.__index__(i, j, k - 1)
                        self.__test_list.add((min(current,bottom),max(current,bottom)))
                    if k < self.height - 1:
                        top = self.__index__(i, j, k + 1)
                        self.__test_list.add((min(current,top),max(current,top)))

        print(self.__test_list)
        
    def __cost__(self,route):
        cost = 0
        for adj in self.__test_list:
            cost = cost + (route[adj[0]] - route[adj[1]])**2
        return cost
        
    def __two_opt__(self,route):
        best = route
        best_cost = self.__cost__(best)
        print(best_cost)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)):
                    if j-i == 1: continue # changes nothing, skip then
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                    if self.__cost__(new_route) < best_cost:  # what should cost be?
                        best = new_route
                        best_cost = self.__cost__(best)
                        print(best_cost)
                        improved = True
        route = best
        return best
   
if __name__ == "main":   
    Opt = OptimizedVolumeMap(65,21,12)
    Opt.Optimize()
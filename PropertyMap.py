from VolumeMap import OptimizedVolumeMap
import numpy as np

class PropertyMap:
    def __init__(self,width,length,height,Map=None,Data=None):
        self.width = width
        self.height = height
        self.length = length
        self.volume = width * length * height
        if Map is None:
            self.__mapping = OptimizedVolumeMap(width,length,height)
        else:
            self.__mapping = Map
        
        if Data is None:
            self.data = np.zeros(self.volume)
        else:
            self.data = np.copy(Data)
    
    def __getitem__(self,coord):
        return self.data[self.__mapping.getMappedIndex(coord)]
        
    def __setitem__(self,coord,val):
        self.data[self.__mapping.getMappedIndex(coord)] = val
    def __str__(self):
        return "{0}".format(self.data)
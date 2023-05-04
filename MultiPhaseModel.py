from scipy.sparse import coo
from VolumeMap import OptimizedVolumeMap
from PropertyMap import PropertyMap
import numpy as np
from scipy.special import expit
import scipy.sparse.linalg
import scipy.sparse as sp
import matplotlib.pyplot as plt
from time import sleep

class TransientConductionModel:
    def __init__(self,width,length,height,CoordinateMap,TransSize = 10, TransSharpness = 1):
        self.TransitionSize = TransSize
        self.width = width
        self.length = length
        self.height = height
        self.volume = width * length * height
        self.__coord_map = CoordinateMap
        self.__prev_step_size = 0
        self.__sharpness = TransSharpness
        #material properties
        self.Density = PropertyMap(width,length,height,CoordinateMap)
        self.ThermalConductivity = PropertyMap(width,length,height,CoordinateMap)        
        self.SolidSpecificHeat = PropertyMap(width,length,height,CoordinateMap)
        self.HeatOfFusion = PropertyMap(width,length,height,CoordinateMap)
        self.LiquidSpecificHeat = PropertyMap(width,length,height,CoordinateMap)
        self.MeltingTemperature = PropertyMap(width,length,height,CoordinateMap)
        #Initial condition
        self.Temperature = PropertyMap(width,length,height,CoordinateMap)
        #Boundary Conditions
        
        self.EastBoundaryTransferCoeficient = np.zeros((self.length,self.height))
        self.EastBoundaryTemperature = np.zeros((self.length,self.height))
        
        self.WestBoundaryTransferCoeficient = np.zeros((self.length,self.height))
        self.WestBoundaryTemperature = np.zeros((self.length,self.height))
        
        self.NorthBoundaryTransferCoeficient = np.zeros((self.width,self.height))
        self.NorthBoundaryTemperature = np.zeros((self.width,self.height))
        
        self.SouthBoundaryTransferCoeficient = np.zeros((self.width,self.height))
        self.SouthBoundaryTemperature = np.zeros((self.width,self.height))
        
        self.TopBoundaryTransferCoeficient = np.zeros((self.width,self.length))
        self.TopBoundaryTemperature = np.zeros((self.width,self.length))
        
        self.BottomBoundaryTransferCoeficient = np.zeros((self.width,self.length))
        self.BottomBoundaryTemperature = np.zeros((self.width,self.length))
        
    def Init(self):
        self.__alfa = PropertyMap(self.width,self.length,self.height,self.__coord_map,self.ThermalConductivity.data / self.Density.data)
        self.__Hsf = PropertyMap(self.width,self.length,self.height,self.__coord_map, self.HeatOfFusion.data / self.__TransArea())

    def __step(self,x):
        if self.__sharpness > 0:
            return expit(x * self.__sharpness)
        else:
            return (x > 0).astype(np.int32)
    
    def __ramp(self,x):
        if self.__sharpness > 0:
            return np.log(1 + np.exp(x * self.__sharpness)) / self.__sharpness
        else:
            return (x > 0).astype(np.int32) * x

    def __TransArea(self):
        if self.__sharpness > 0:
            x = 10 / self.__sharpness
            res = self.__ramp(x + self.TransitionSize) - self.__ramp(x)
            return res
        else: return self.TransitionSize
    
    def __TempCorrection(self, T):
        Amplitude = self.TransitionSize / self.__TransArea()
        Excess = T.data - self.MeltingTemperature.data
        return Amplitude * (self.__ramp(Excess + self.TransitionSize) - self.__ramp(Excess))

    def __GetMeltingPropertyCoeficients(self,T):
            
        T2 = T.data - self.MeltingTemperature.data
        T1 = T2 + self.TransitionSize
        C1 = self.__step(T1)
        C2 = self.__step(T2)
        Cs = PropertyMap(self.width,self.length,self.height,self.__coord_map,1 - C1)
        Csf = PropertyMap(self.width,self.length,self.height,self.__coord_map,C1 - C2)
        Cl = PropertyMap(self.width,self.length,self.height,self.__coord_map,C2)
        return Cs, Csf, Cl
        
    def __SpecificHeat(self,Cs, Csf, Cl):
        
        Cps = Cs.data * self.SolidSpecificHeat.data
        Hsf = Csf.data * self.__Hsf.data
        Cpl = Cl.data * self.LiquidSpecificHeat.data
        Cp = Cps + Hsf + Cpl
        return PropertyMap(self.width,self.length,self.height,self.__coord_map,Cp)
        
    def __SpecificHeat_Residual(self, Cs, Csf, Cl):
        Hsf = Csf.data * (self.MeltingTemperature.data - self.TransitionSize) * (self.SolidSpecificHeat.data - self.__Hsf.data)
        Cpl = Cl.data * ((self.MeltingTemperature.data * (self.SolidSpecificHeat.data - self.LiquidSpecificHeat.data )) - (self.TransitionSize *  self.SolidSpecificHeat.data) + self.HeatOfFusion.data)
        Cp = Hsf + Cpl
        return PropertyMap(self.width,self.length,self.height,self.__coord_map,Cp)
        
    def __Temperature_Residual(self, Cs, Csf, Cl):
        return PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Hsf = Csf.data * (self.MeltingTemperature.data - self.TransitionSize) * (self.SolidSpecificHeat.data / self.__Hsf.data - 1)
        Ratio = self.SolidSpecificHeat.data / self.LiquidSpecificHeat.data
        Cpl = Cl.data * ((self.MeltingTemperature.data * (Ratio - 1)) - (self.TransitionSize * Ratio) + self.HeatOfFusion.data / self.LiquidSpecificHeat.data)
        Cp = Hsf + Cpl
        return PropertyMap(self.width,self.length,self.height,self.__coord_map,Cp)

    def __CaracteristicLengths(self, Width, Length, Height):
        Least = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Lwest = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        
        Lnorth = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Lsouth = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        
        Ltop = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Lbottom = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        for i in range(self.width):
            for j in range(self.length):
                for k in range(self.height):
                    w =  Width[i, j, k]
                    l = Length[i, j, k]
                    h = Height[i, j, k]

                    Ayz = l * h
                    if i < self.width - 1: Least[i, j, k] = Ayz * 2 / (w + Width[i + 1, j, k])
                    if i > 0: Lwest[i, j, k] = Ayz * 2 / (w + Width[i - 1, j, k])
                    
                    Axz = w * h
                    if j < self.length - 1: Lnorth[i, j, k] = Axz * 2 / (l + Length[i, j + 1, k])
                    if j > 0: Lsouth[i, j, k] = Axz * 2 / (l + Length[i, j - 1, k])
                    
                    Axy = w * l
                    if k < self.height - 1: Ltop[i, j, k] = Axy * 2 / (h + Height[i, j, k + 1])
                    if k > 0: Lbottom[i, j, k] = Axy * 2 / (h + Height[i, j, k - 1])
        return Least, Lwest, Lnorth, Lsouth, Ltop, Lbottom
        
    def __CellCoeficients(self, Least, Lwest, Lnorth, Lsouth, Ltop, Lbottom):
        Reast = PropertyMap(self.width,self.length,self.height,self.__coord_map, Lwest.data * self.__alfa.data)
        Rwest = PropertyMap(self.width,self.length,self.height,self.__coord_map, Least.data * self.__alfa.data)

        Rnorth = PropertyMap(self.width,self.length,self.height,self.__coord_map, Lsouth.data * self.__alfa.data)
        Rsouth = PropertyMap(self.width,self.length,self.height,self.__coord_map, Lnorth.data * self.__alfa.data)

        Rtop = PropertyMap(self.width,self.length,self.height,self.__coord_map, Lbottom.data * self.__alfa.data)
        Rbottom = PropertyMap(self.width,self.length,self.height,self.__coord_map, Ltop.data * self.__alfa.data)

        Ceast = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Cwest = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        
        Cnorth = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Csouth = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        
        Ctop = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Cbottom = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        
        for i in range(self.width):
            for j in range(self.length):
                for k in range(self.height):
                    if i < self.width - 1: Ceast[i, j, k] = - Reast[i + 1, j, k]
                    if i > 0: Cwest[i, j, k] = - Rwest[i - 1, j, k]
                    
                    if j < self.length - 1: Cnorth[i,j,k] = - Rnorth[i, j + 1, k]
                    if j > 0: Csouth[i, j, k] = - Rsouth[i, j - 1, k]
                    
                    if k < self.height - 1: Ctop[i, j, k] = - Rtop[i, j, k + 1]
                    if k > 0: Cbottom[i, j, k] = - Rbottom[i, j, k - 1]
        return Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom
        
    def __BoundaryCoeficients(self, Width, Length, Height, stepSize):
        def Coef(face, coord, h):
            face_coord = ()
            A = 0
            L = 0
            if face == 'yz':
                face_coord = (coord[1], coord[2])
                A = Length[coord] * Height[coord]
                L = Width[coord]
            elif face == 'xz':
                face_coord = (coord[0], coord[2])
                A = Width[coord] * Height[coord]
                L = Length[coord]
            else:
                face_coord = (coord[0], coord[1])
                A = Width[coord] * Length[coord]
                L = Height[coord]
            V = A * L
            k = self.ThermalConductivity[coord] * 4
            _h = h[face_coord]
            #   A * h * k * V
            #-----------------------
            #(h * p * V * L) + (A * k)
            N = A * _h * k * V
            D = _h * self.Density[coord] * V * L + A * k

            return - N / D
        
        Cboundary = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        Rboundary = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        #East + West Boundary
        for j in range(self.length):
            for k in range(self.height):       
                East = Coef('yz', (self.width - 1, j, k), self.EastBoundaryTransferCoeficient)
                Cboundary[self.width - 1, j, k] -= East
                Rboundary[self.width - 1, j, k] -= East * self.EastBoundaryTemperature[j, k]
                West = Coef('yz', (0, j, k), self.WestBoundaryTransferCoeficient)
                Cboundary[0, j, k] -= West
                Rboundary[0, j, k] -= West * self.WestBoundaryTemperature[j, k]
                

        for i in range(self.width):
            for k in range(self.height):
                North = Coef('xz', (i, self.length - 1, k), self.NorthBoundaryTransferCoeficient)
                Cboundary[i, self.length - 1, k] -= North
                Rboundary[i, self.length - 1, k] -= North * self.NorthBoundaryTemperature[i, k]
                South = Coef('xz', (i, 0, k), self.SouthBoundaryTransferCoeficient)
                Cboundary[i, 0, k] -= South
                Rboundary[i, 0, k] -= South * self.SouthBoundaryTemperature[i, k]
                

        for i in range(self.width):
            for j in range(self.length):
                Top = Coef('xy', (i, j, self.height - 1), self.TopBoundaryTransferCoeficient)
                Cboundary[i, j, self.height - 1] -= Top
                Rboundary[i, j, self.height - 1] -= Top * self.TopBoundaryTemperature[i, j]
                Bottom = Coef('xy', (i, j, 0), self.BottomBoundaryTransferCoeficient)
                Cboundary[i, j, 0] -= Bottom
                Rboundary[i, j, 0] -= Bottom * self.BottomBoundaryTemperature[i, j]
                

        return Cboundary, Rboundary

    
    def __MultWithNeigbour(self,neigbour,OwnValue,NeigbourValue):
        Result = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        #print(Result.data)
        #print(OwnValue.data)
        #print(NeigbourValue.data)
        Offsets = {
            'e': [( 1,  0,  0),( 0,  0,  0),(-1,  0,  0)],
            'w': [(-1,  0,  0),( 1,  0,  0),( 0,  0,  0)],
            'n': [( 0,  1,  0),( 0,  0,  0),( 0, -1,  0)],
            's': [( 0, -1,  0),( 0,  1,  0),( 0,  0,  0)],
            't': [( 0,  0,  1),( 0,  0,  0),( 0,  0, -1)],
            'b': [( 0,  0, -1),( 0,  0,  1),( 0,  0,  0)]
        }
        offset = Offsets[neigbour]
        for i in range(offset[1][0],self.width + offset[2][0]):
            for j in range(offset[1][1], self.length + offset[2][1]):
                for k in range(offset[1][2], self.height + offset[2][2]):
                    Result[i, j, k] = OwnValue[i, j, k] * NeigbourValue[i + offset[0][0], j + offset[0][1], k + offset[0][2]]
        return Result
        
    def __ConductionResidual(self, Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom, T):
        #print(T.data)
        ConductionResidual = self.__MultWithNeigbour('e',Ceast,T)
        #print(ConductionResidual.data)
        ConductionResidual.data += self.__MultWithNeigbour('w',Cwest,T).data
        
        ConductionResidual.data += self.__MultWithNeigbour('n',Cnorth,T).data
        ConductionResidual.data += self.__MultWithNeigbour('s',Csouth,T).data
        
        ConductionResidual.data += self.__MultWithNeigbour('t',Ctop,T).data
        ConductionResidual.data += self.__MultWithNeigbour('b',Cbottom,T).data
        return ConductionResidual

    def __IsUnstable(self,Temps):
        count = 0
        for j in range(self.length):
            for k in range(self.height):
                prevAvg = Temps[0, j, k]
                prevTemp = Temps[0, j, k]
                for i in range(1,self.width):
                    T = Temps[i, j, k]                   
                    if (abs(T - prevTemp) > 0.01) & ((prevAvg < prevTemp > T) | (prevAvg > prevTemp < T)):
                        count += 1
                        if count > 1: return True
                    elif count > 0:
                        count -= 1
                    prevAvg = prevTemp
                    prevTemp = T

        for i in range(self.width):
            for k in range(self.height):
                prevAvg = Temps[i, 0, k]
                prevTemp = Temps[i, 0, k]
                for j in range(1,self.length):
                    T = Temps[i, j, k]                   
                    if (abs(T - prevTemp) > 0.01) & ((prevAvg < prevTemp > T) | (prevAvg > prevTemp < T)):
                        count += 1
                        if count > 1: return True
                    elif count > 0:
                        count -= 1
                    prevAvg = prevTemp
                    prevTemp = T
        
        for i in range(self.width):
            for j in range(self.length):
                prevAvg = Temps[i, j, 0]
                prevTemp = Temps[i, j, 0]
                for k in range(1,self.height):
                    T = Temps[i, j, k]                   
                    if (abs(T - prevTemp) > 0.01) & ((prevAvg < prevTemp > T) | (prevAvg > prevTemp < T)):
                        count += 1
                        if count > 1: return True
                    elif count > 0:
                        count -= 1
                    prevAvg = prevTemp
                    prevTemp = T
        return count > 1

        
    def __DoIteration(self,Sys,Res,T, VolSpeed, Ccond, Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom):
        Cs, Csf, Cl = self.__GetMeltingPropertyCoeficients(T)
        SysTrans = self.__SpecificHeat(Cs, Csf, Cl)
        SysTrans.data = SysTrans.data * VolSpeed.data
        #print(SysTrans)
        #print(VolSpeed)
        B = self.__SpecificHeat_Residual(Cs, Csf, Cl)
        C = self.__Temperature_Residual(Cs, Csf, Cl)
        #print(Cs.data)
        #print(Csf.data)
        #print(Cl.data)
        #print(C.data)
        TransientEnergyLag = PropertyMap(self.width,self.length,self.height,self.__coord_map, C.data * Ccond.data)
        #print(TransientEnergyLag.data[self.width - 1])
        TransientAbsorptionResidual = PropertyMap(self.width,self.length,self.height,self.__coord_map, B.data * VolSpeed.data)
        #print(TransientAbsorptionResidual.data)
        TransientConductionResidual = self.__ConductionResidual(Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom, C)
        #print(TransientConductionResidual.data)
        ##b
        Residual = Res.data - TransientAbsorptionResidual.data - TransientEnergyLag.data - TransientConductionResidual.data
        #print(Residual)
        ##A
        System = Sys + sp.diags(SysTrans.data,format='csr')
        #print(np.array(
        #    [[System[self.width - 2, self.width - 2],System[self.width - 2, self.width - 1], Residual[self.width - 2]],
        #    [System[self.width - 1, self.width - 2],System[self.width - 1, self.width - 1], Residual[self.width - 1]]] ) )

        #print(System.toarray())
        #Tdebug = np.full(self.volume, 263)

        #Tdebug = System.dot(Tdebug)
        #print(Tdebug)
        #Tdebug = Tdebug - Residual
        #print(Tdebug)
        Tnext = sp.linalg.spsolve(System,Residual)
    
        return PropertyMap(self.width,self.length,self.height,self.__coord_map, Tnext)
    
    def __NextTimeStep(self,stepSize, WidthsMap, LengthsMap, HeightsMap, HeatInputMap):
        if stepSize != self.__prev_step_size :
            self.__prev_step_size = stepSize
            print("Step Size = {0}s".format(stepSize))
        #Debug = np.array([-0.32773846,  0.09710769,  0.09710769,  0.09710769,  0.09710769,  0.09710769, 0.09710769 , 0.09710769,  0.09710769,  0.61906154])
        Volume = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        HeatGeneration = PropertyMap(self.width,self.length,self.height,self.__coord_map)
        for i in range(self.width):
            for j in range(self.length):
                for k in range(self.height):
                    vol = WidthsMap[i, j, k] * LengthsMap[i, j, k] * HeightsMap[i, j, k]
                    HeatGeneration[i, j, k] = HeatInputMap[i, j, k]
                    Volume[i, j, k] = vol
        #print(max(HeatGeneration.data))
        VolSpeed = PropertyMap(self.width,self.length,self.height,self.__coord_map,Volume.data / stepSize)
        Least, Lwest, Lnorth, Lsouth, Ltop, Lbottom = self.__CaracteristicLengths(WidthsMap, LengthsMap, HeightsMap)
        
        Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom = self.__CellCoeficients(Least, Lwest, Lnorth, Lsouth, Ltop, Lbottom)
        #print(Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom)
        Cboundary, RBoundary = self.__BoundaryCoeficients(WidthsMap, LengthsMap, HeightsMap, stepSize)
        #print(Debug/Cboundary.data)
        SteadyConductionCoeficient = PropertyMap(self.width,self.length,self.height,self.__coord_map,self.__alfa.data * (Least.data + Lwest.data + Lnorth.data + Lsouth.data + Ltop.data + Lbottom.data))
        #print(SteadyConductionCoeficient)
        #print(Debug/SteadyConductionCoeficient.data)
        Cs, Csf, Cl = self.__GetMeltingPropertyCoeficients(self.Temperature)
        #print(Cboundary.data)
        AbsorptionResidual = PropertyMap(self.width,self.length,self.height,self.__coord_map, (self.__SpecificHeat(Cs, Csf, Cl).data * self.Temperature.data + self.__SpecificHeat_Residual(Cs, Csf, Cl).data) * VolSpeed.data)

        SteadyResidual = PropertyMap(self.width,self.length,self.height,self.__coord_map, AbsorptionResidual.data + RBoundary.data + HeatGeneration.data)
        #print(AbsorptionResidual.data)
        #print(BoundryResidual.data)
        Ccond = PropertyMap(self.width,self.length,self.height,self.__coord_map,SteadyConductionCoeficient.data + Cboundary.data)

        SteadySystem = sp.lil_matrix((self.volume,self.volume))
        for i in range(self.width):
            for j in range(self.length):
                for k in range(self.height):
                    index = self.__coord_map.getMappedIndex((i, j, k))
                    SteadySystem[index, index] = Ccond[i, j, k]
                    if i < self.width - 1 : 
                        eastindex = self.__coord_map.getMappedIndex((i + 1, j, k))
                        SteadySystem[index,eastindex] = Ceast[i, j, k]
                    if i > 0 :
                        westindex = self.__coord_map.getMappedIndex((i - 1, j, k))
                        SteadySystem[index,westindex] = Cwest[i, j, k]
                    if j < self.length - 1:
                        northindex = self.__coord_map.getMappedIndex((i, j + 1, k))
                        SteadySystem[index, northindex] = Cnorth[i, j, k]
                    if j > 0:
                        southindex = self.__coord_map.getMappedIndex((i, j - 1, k))
                        SteadySystem[index, southindex] = Csouth[i, j, k]
                    if k < self.height - 1:
                        topindex = self.__coord_map.getMappedIndex((i, j, k + 1))
                        SteadySystem[index, topindex] = Ctop[i, j, k]
                    if k > 0:
                        bottomindex = self.__coord_map.getMappedIndex((i, j, k - 1))
                        SteadySystem[index, bottomindex] = Cbottom[i, j, k]
        SteadySystem = SteadySystem.tocsr()
        Titer = PropertyMap(self.width,self.length,self.height,self.__coord_map, self.Temperature.data.copy())
        i = 0
        for i in range(100):
            Tnext = self.__DoIteration(SteadySystem,SteadyResidual, Titer, VolSpeed, Ccond, Ceast, Cwest, Cnorth, Csouth, Ctop, Cbottom)
            err = np.square(Tnext.data - Titer.data).mean()
            Titer = Tnext
            #print(err)
            if err < 0.0001:
                #print(i)
                break
        #print(i)
        Unstable = self.__IsUnstable(Titer)
        if Unstable:
            print("Unstable")
            sleep(3600)
        if i == 99:
            print("Non Convergent")
            return Titer
            
        if (i == 99) | Unstable:
            #self.__show_plot(Titer, WidthsMap,LengthsMap, HeightsMap)
            
            for i in range(10):
                Titer = self.__NextTimeStep(stepSize/10,WidthsMap,LengthsMap, HeightsMap, HeatInputMap)
        #self.__show_plot(self.Temperature, WidthsMap,LengthsMap, HeightsMap)
        self.Temperature = Titer
        return Titer

    def NextTimeStep(self,stepSize, WidthsMap, LengthsMap, HeightsMap, HeatInputMap):
        Res = self.__NextTimeStep(stepSize, WidthsMap, LengthsMap, HeightsMap, HeatInputMap)
        Res = PropertyMap(self.width,self.length,self.height,self.__coord_map, Res.data)# - self.__TempCorrection(Res))
        return Res
        
    def __show_plot(self,Temperatures, WidthsMap, LengthsMap, HeightsMap):
        X = np.zeros((self.width,self.length, self.height))
        Y = np.zeros((self.width,self.length, self.height))
        Z = np.zeros((self.width,self.length, self.height))

        X[0, 0, 0] = 0
        Y[0, 0, 0] = 0
        Z[0, 0, 0] = 0
        Temps = np.zeros((self.width,self.length, self.height))
        for i in range(self.width):
            for j in range(self.length):
                for k in range(self.height):
                    Temps[i, j, k] = Temperatures[i, j, k]
                    if i > 0:
                        X[i, j, k] = WidthsMap[i, j, k] + X[i - 1, j, k] 
                    if j > 0:       
                        Y[i, j, k] = LengthsMap[i, j, k] + Y[i, j - 1, k]   
                    if k > 0:    
                        Z[i, j, k] = HeightsMap[i, j, k] + Z[i, j, k - 1]
        for k in range(self.height):
            #print(Temps[:,:,k])
            plt.plot(X[:,0,0],Temps[:,0,0])
        
        plt.show()
        
    
        
        
        
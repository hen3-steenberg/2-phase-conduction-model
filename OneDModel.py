
from MultiPhaseModel import TransientConductionModel
from VolumeMap import OptimizedVolumeMap
import numpy as np

class OneDimensionalProplem:
    def __init__(self, Width, Length, Height, NumCells, Cs, Hsf, Cl, k, p, hboundary, Tmelt, Tinitial, Tsurr, Ttrans, Q, sharpness):
        """
        Model Geometery
        """
        self.Width = Width
        self.Length = Length
        self.Height = Height
        self.XCells = NumCells
        self.YCells = 1
        self.ZCells = 1
        self.NumCells = NumCells
        """
        Material properties
        """
        self.Cs = Cs
        self.Hsf = Hsf
        self.Cl = Cl
        self.k = k
        self.p = p
        self.Tmelt = Tmelt
        """
        Initial Conditions
        """
        self.Tinitial = Tinitial
        """
        Boundary conditions
        """
        self.Tsurr = Tsurr
        self.h = hboundary
        self.Q = Q
        """
        Initializing model
        """
        self.Map = OptimizedVolumeMap(self.XCells,self.YCells,self.ZCells)
        self.Model = TransientConductionModel(self.XCells,self.YCells,self.ZCells,self.Map,Ttrans, sharpness)
        """
        Setting Model Properties
        """
        for i in range(self.XCells):
            for j in range(self.YCells):
                for k in range(self.ZCells):
                    """
                    Setting material properties
                    """
                    self.Model.Density[i, j, k] = self.p
                    self.Model.ThermalConductivity[i, j, k] = self.k
                    self.Model.SolidSpecificHeat[i, j, k] = self.Cs
                    self.Model.HeatOfFusion[i, j, k] = self.Hsf
                    self.Model.LiquidSpecificHeat[i, j, k] = self.Cl
                    self.Model.MeltingTemperature[i, j, k] = self.Tmelt
                    """
                    Setting Initial conditions
                    """
                    self.Model.Temperature[i, j, k] = Tinitial
        """
        Setting boundary conditions
        """
        for j in range(self.YCells):
            for k in range(self.ZCells):
                self.Model.EastBoundaryTransferCoeficient[j, k] = self.h
                self.Model.EastBoundaryTemperature[j, k] = self.Tinitial

                self.Model.WestBoundaryTransferCoeficient[j, k] = self.h
                self.Model.WestBoundaryTemperature[j, k] = self.Tsurr
        for i in range(self.XCells):
            for k in range(self.ZCells):
                self.Model.NorthBoundaryTransferCoeficient[i, k] = self.h
                self.Model.NorthBoundaryTemperature[i, k] = self.Tsurr

                self.Model.SouthBoundaryTransferCoeficient[i, k] = self.h
                self.Model.SouthBoundaryTemperature[i, k] = self.Tsurr
        
        for i in range(self.XCells):
            for j in range(self.YCells):
                self.Model.TopBoundaryTransferCoeficient[i, j] = self.h
                self.Model.TopBoundaryTemperature[i, j] = self.Tsurr

                self.Model.BottomBoundaryTransferCoeficient[i, j] = self.h
                self.Model.BottomBoundaryTemperature[i, j] = self.Tsurr
        """
        Setting Up Geometry variables
        """
        self.Widths = np.full((self.XCells,self.YCells,self.ZCells),self.Width/self.XCells)
        self.Lengths = np.full((self.XCells,self.YCells,self.ZCells),self.Length/self.YCells)
        self.Heights = np.full((self.XCells,self.YCells,self.ZCells),self.Height/self.ZCells)
        self.Xcoords = np.arange(0,self.Width, self.Width/self.XCells)
        """
        Setting up Heat Generation
        """
        self.Heat = np.zeros((self.XCells,self.YCells,self.ZCells))

        GenCells = list(range(int(0.4 * self.XCells), int(0.6 * self.XCells) + 1))
        print(len(GenCells))
        for i in GenCells:
            self.Heat[i, 0, 0] = self.Q / len(GenCells)
        #self.Heat[0, 0, 0] = self.Q
        self.Model.Init()

    def AdvanceModel(self, TimeStep):
        Temps = self.Model.NextTimeStep(TimeStep, self.Widths, self.Lengths, self.Heights, self.Heat)
        return Temps.data, self.Xcoords
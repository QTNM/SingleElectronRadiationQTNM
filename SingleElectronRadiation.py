import numpy
import matplotlib.pyplot as pyplot


# Constants
ECharge = 1.602176634*10**-19 # Coulombs
pi = numpy.pi
VacPerm = 8.8541878128*10**-12	# F/m - Vacuum permittivity
c = 299792458 # Speed of light in m/s

# Initialize Variables
DetectorPosition = numpy.array([0,0,0])
EVelocity = numpy.array([0,0,0]) # Electron velocity at retarded time
EAcceleration = numpy.array([0,0,0]) # Electron acceleration at retarded time
EPosition = numpy.array([0,0,0])


# Function definitions

def CalcNonRelEField(Position,Time,EPosition,EAcceleration):
    # Calculates the non-relativistic electric field at a given position and time
    # for an electron located at EPosition, with acceleration given.
    # This formula assumes non relativistic speeds so the emission point = the position of the electron.
    
    # First calculate vector/unit vector between position and emission point
    r = numpy.linalg.norm(Position-EPosition) # Distance from electron (radiation point) to detector
    VecEmissionToObsUnit = (Position-EPosition)/r # Known as "n" in the formulae
    n = VecEmissionToObsUnit # Convenience
    # Calculate EField from non-rel E-Field formula
    EField = ECharge/(4*pi*VacPerm*c)*(numpy.cross(n,numpy.cross(n,EAcceleration)/c)/r)
    return EField


def CalcElectronCircularMotion(Time,AngFreq,Amplitude,ZPos):
    # Circular Motion around Z axis at Z
    EPosition = Amplitude*numpy.array([numpy.cos(AngFreq*Time),numpy.sin(AngFreq*Time),ZPos])
    EVelocity = Amplitude*AngFreq*numpy.array([-numpy.sin(AngFreq*Time),numpy.cos(AngFreq*Time),0])
    EAcceleration = Amplitude*AngFreq**2*numpy.array([-numpy.cos(AngFreq*Time),-numpy.sin(AngFreq*Time),0])
    return EPosition,EVelocity,EAcceleration


def CreateDetectorArray(Resolution,x0,x1,y0,y1,z):
    # Create an array with an xy spatial resolution of Resolution at position z
    # Resolution = The number of points between e.g. -5 and 5
    xs = numpy.linspace(x0,x1,Resolution)
    ys = numpy.linspace(y0,y1,Resolution)
    ZZ = numpy.zeros((Resolution,Resolution))+z
    XX,YY = numpy.meshgrid(xs,ys)
    DetectorPositionArray = numpy.dstack([XX,-YY,ZZ]).reshape(Resolution,Resolution,3)
    return DetectorPositionArray


def CalcNonRelEFieldArray(PositionArray,Time,EPosition,EAcceleration):
    EFieldArray = numpy.zeros((Resolution,Resolution,3))
    for i in range(Resolution):
        for j in range(Resolution):
            EFieldVector = CalcNonRelEField(DetectorPositionArray[i][j],Time,EPosition,EAcceleration)
            EFieldArray[i,j] = EFieldVector
    return EFieldArray


def CalcPoyntingVectorMagnitude(EFieldMagnitudes):
    PoyntingMagnitudes = VacPerm*c*EFieldMagnitudes**2
    return PoyntingMagnitudes


def FindColourBounds(InputArray):
    ColourMax = numpy.max(InputArray[0])
    ColourMin = numpy.min(InputArray[0])
    for i in range(TimeResolution):
        NewMax = numpy.max(InputArray[i])
        NewMin = numpy.min(InputArray[i])
        if ColourMax < NewMax:
            ColourMax = NewMax
        if ColourMin > NewMin:
            ColourMin = NewMin
    return(ColourMin,ColourMax)



# Run the code
if __name__ == "__main__":
    # Define the detector's position
    Time = 0
    Resolution = 50
    x0, x1 = -0.025,0.025 # m
    y0, y1 = -0.01,0.01 # m
    z = 0.05 # m
    DetectorPositionArray = CreateDetectorArray(Resolution,x0,x1,y0,y1,z)
    
    # Calculate non relativistic E Field for a range of points (a square detector centred on the z axis)
    # Then calculate the E Field for each of these points and store it in an array
    EFieldArray = CalcNonRelEFieldArray(DetectorPositionArray,Time,EPosition,EAcceleration) # EFieldArray[xindex, yindex]
    EFieldArrayXComponents = EFieldArray[0:,0:,0]
    EFieldArrayYComponents = EFieldArray[0:,0:,1]
    EFieldArrayZComponents = EFieldArray[0:,0:,2]
    EFieldMagnitudes = numpy.linalg.norm(EFieldArray,axis=(2))
    
    Time=0
    AngFreq = 1.7563*10**11 # rad/s
    Amplitude = 0.000448 # m
    ZPos = 0
    EPosition, EVelocity, EAcceleration = CalcElectronCircularMotion(Time,AngFreq,Amplitude,ZPos)
    
    EFieldArray = CalcNonRelEFieldArray(DetectorPositionArray,Time,EPosition,EAcceleration) # EFieldArray[xindex, yindex]
    EFieldArrayXComponents = EFieldArray[0:,0:,0]
    EFieldArrayYComponents = EFieldArray[0:,0:,1]
    EFieldArrayZComponents = EFieldArray[0:,0:,2]
    EFieldMagnitudes = numpy.linalg.norm(EFieldArray,axis=(2))
    PoyntingMagnitudes = CalcPoyntingVectorMagnitude(EFieldMagnitudes)
    
    # Create animation along path
    TimeResolution = 15
    MaxTime = 2*pi/AngFreq # For circular motion
    TimeStep = MaxTime/TimeResolution

    ColourMaxEMag = -999999
    ColourMinEMag = +999999
    EFieldXs = numpy.zeros(TimeResolution, dtype=object)
    EFieldYs = numpy.zeros(TimeResolution, dtype=object)
    EFieldZs = numpy.zeros(TimeResolution, dtype=object)
    EFieldMagnitudesArray = numpy.zeros(TimeResolution, dtype= object)
    PoyntingMagnitudesArray = numpy.zeros(TimeResolution, dtype=object)
    for i in range(TimeResolution):
        # Calculate electron motion for each time step
        EPosition, EVelocity, EAcceleration = CalcElectronCircularMotion(Time,AngFreq,Amplitude,ZPos)
        
        # Calculate EField and Poynting for each time step
        EFieldArray = CalcNonRelEFieldArray(DetectorPositionArray,Time,EPosition,EAcceleration)
        EFieldArrayXComponents = EFieldArray[0:,0:,0]
        EFieldArrayYComponents = EFieldArray[0:,0:,1]
        EFieldArrayZComponents = EFieldArray[0:,0:,2]
        EFieldMagnitudes = numpy.linalg.norm(EFieldArray,axis=(2))    
        PoyntingMagnitudes = CalcPoyntingVectorMagnitude(EFieldMagnitudes)
        
        EFieldXs[i] = EFieldArrayXComponents
        EFieldYs[i] = EFieldArrayYComponents
        EFieldZs[i] = EFieldArrayZComponents
        EFieldMagnitudesArray[i] = EFieldMagnitudes
        PoyntingMagnitudesArray[i] = PoyntingMagnitudes
        
        Time = Time + TimeStep
    # end loop
    
    # Calculate Power from Poynting vector
    PixelArea = (x1-x0)/Resolution*(y1-y0)/Resolution
    PowerArray = PoyntingMagnitudesArray*PixelArea
        
    # Set colour bar graph limits
    ColourMinEMag,ColourMaxEMag = FindColourBounds(EFieldMagnitudesArray)
    ColourMinPoynting,ColourMaxPoynting = FindColourBounds(PoyntingMagnitudesArray)
    ColourMinPowerArray,ColourMaxPowerArray = FindColourBounds(PowerArray)
     
    for i in range(TimeResolution):
        # Plot and output images
        figsize = [7, 9]     # figure size, inches
        fig, ax = pyplot.subplots(nrows=3, ncols=1, figsize=figsize)
        graph0 = ax[0].imshow(EFieldMagnitudesArray[i], extent=(x0,x1,y0,y1))
        graph1 = ax[1].imshow(PoyntingMagnitudesArray[i], extent=(x0,x1,y0,y1))
        graph2 = ax[2].imshow(PowerArray[i], extent=(x0,x1,y0,y1))
        pyplot.xlabel("x (m)")
        pyplot.ylabel("y (m)")
        pyplot.colorbar(graph0, ax=ax[0], orientation="vertical")
        pyplot.colorbar(graph1, ax=ax[1], orientation="vertical")
        pyplot.colorbar(graph2, ax=ax[2], orientation="vertical")
        graph0.set_clim(ColourMinEMag,ColourMaxEMag)
        graph1.set_clim(ColourMinPoynting,ColourMaxPoynting)
        graph2.set_clim(ColourMinPowerArray,ColourMaxPowerArray)
        ax[0].set_title("E Field Magnitude")
        ax[1].set_title("Poynting Vector Magnitudes")
        ax[2].set_title("Power (W)")
        
        pyplot.savefig("Images\Poynting%s" % i)
    # end loop


    
    
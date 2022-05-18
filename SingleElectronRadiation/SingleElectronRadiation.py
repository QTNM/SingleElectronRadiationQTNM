import numpy
import matplotlib.pyplot as pyplot

#pyplot.rcdefaults()
pyplot.rcParams.update({'font.size': 18})


# Constants
ECharge = 1.602176634e-19 # Coulombs
pi = numpy.pi
VacPerm = 8.8541878128e-12	# F/m - Farads per metre - Vacuum permittivity
c = 299792458 # Speed of light in m/s
EMass = 9.10938356e-31 # kg

# Initialize Variables
DetectorPosition = numpy.array([0,0,0])
EVelocity = numpy.array([0,0,0]) # Electron velocity at retarded time
EAcceleration = numpy.array([0,0,0]) # Electron acceleration at retarded time
EPosition = numpy.array([0,0,0])
EPositionRet = numpy.array([0,0,0]) # Electron retarted position (Electron position evaluated at retarded time)
EEnergy = 18600*ECharge # Electron Energy in Joules, 18.6keV # Initial Energy


# Function definitions

def CalcNonRelEField(Position,Time,EPosition,EAcceleration):
    # Calculates the non-relativistic electric field at a given position and time
    # for an electron located at EPosition, with acceleration given.
    # This formula assumes non relativistic speeds so the emission point = the position of the electron.
    # First calculate vector/unit vector between position and emission point
    r = numpy.linalg.norm(Position-EPosition) # Distance from electron (radiation point) to detector
    VecEmissionToObsUnit = (Position-EPosition)/r # Unit vector in direction from emission point to detector position
    n = VecEmissionToObsUnit
    # Calculate EField from non-rel E-Field formula
    EField = ECharge/(4*pi*VacPerm*c)*(numpy.cross(n,numpy.cross(n,EAcceleration)/c)/r)
    return EField

def CalcNonRelNearEField(Position,Time,EPosition,EVelocity,EAcceleration):
    # Calculates the near field non-relativistically
    r = numpy.linalg.norm(Position-EPosition) # Distance from electron (radiation point) to detector
    VecEmissionToObsUnit = (Position-EPosition)/r # Unit vector in direction from emission point to detector position
    n = VecEmissionToObsUnit
    EFieldNear = ECharge/(4*pi*VacPerm) * (n/r**2) # I need a vector
    return EFieldNear

def CalcElectronCircularMotion(Time,AngFreq,Amplitude,ZPos):
    # Circular Motion around Z axis at Zpos
    EPosition = Amplitude*numpy.array([numpy.cos(AngFreq*Time),numpy.sin(AngFreq*Time),ZPos])
    EVelocity = Amplitude*AngFreq*numpy.array([-numpy.sin(AngFreq*Time),numpy.cos(AngFreq*Time),0])
    EAcceleration = Amplitude*AngFreq**2*numpy.array([-numpy.cos(AngFreq*Time),-numpy.sin(AngFreq*Time),0])
    return EPosition,EVelocity,EAcceleration

def CalcElectronCircularMotionPerp(Time,AngFreq,Amplitude,XPos):
    # Circular Motion around x axis
    EPosition = Amplitude*numpy.array([XPos,numpy.sin(AngFreq*Time),numpy.cos(AngFreq*Time)])
    EVelocity = Amplitude*AngFreq*numpy.array([0,numpy.cos(AngFreq*Time),-numpy.sin(AngFreq*Time)])
    EAcceleration = Amplitude*AngFreq**2*numpy.array([0,-numpy.sin(AngFreq*Time),-numpy.cos(AngFreq*Time)])
    return EPosition,EVelocity,EAcceleration

def CreateDetectorArray(Resolution,x0,x1,y0,y1,z):
    # Create an array with an xy spatial resolution of Resolution at position z
    # Resolution = The number of vectors between e.g. -5 and 5
    xs = numpy.linspace(x0,x1,Resolution)
    ys = numpy.linspace(y0,y1,Resolution)
    ZZ = numpy.zeros((Resolution,Resolution))+z
    XX,YY = numpy.meshgrid(xs,ys)
    DetectorPositionArray = numpy.dstack([XX,-YY,ZZ]).reshape(Resolution,Resolution,3) # Access by DetectorPositionArray[y index][x index]
    #print(DetectorPositionArray) # [0][0] gives top left corner [y][x]
    return DetectorPositionArray

def CalcNonRelEFieldArray(DetectorPositionArray,Time,EPosition,EAcceleration,Resolution):
    EFieldArray = numpy.zeros((Resolution,Resolution,3))
    for i in range(Resolution):
        for j in range(Resolution):
            EFieldVector = CalcNonRelEField(DetectorPositionArray[i][j],Time,EPosition,EAcceleration)
            EFieldArray[i,j] = EFieldVector
    return EFieldArray

def CalcNonRelNearEFieldArray(DetectorPositionArray,Time,EPosition,EVelocity,EAcceleration,Resolution):
    NearEFieldArray = numpy.zeros((Resolution,Resolution,3))
    for i in range(Resolution):
        for j in range(Resolution):
            NearEFieldVector = CalcNonRelNearEField(DetectorPositionArray[i][j],Time,EPosition,EVelocity,EAcceleration)
            NearEFieldArray[i,j] = NearEFieldVector
    return NearEFieldArray    

def CalcPoyntingVectorMagnitude(EFieldMagnitudes):
    PoyntingMagnitudes = VacPerm*c*EFieldMagnitudes**2
    return PoyntingMagnitudes

def FindColourBounds(InputArray,TimeResolution):
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
  
def CalcLarmorPower(EAcceleration):
    return ECharge**2 * (numpy.linalg.norm(EAcceleration))**2 / (6 * pi * VacPerm * c**3)
    # return 2/3 * ECharge**2 * (numpy.linalg.norm(EAcceleration))**2  / 4 / pi / VacPerm / c**3

def CalcRelativisticLarmorPower(EVelocity, EAcceleration):
    gamma = 1/numpy.sqrt(1-numpy.dot(EVelocity,EVelocity)/c**2)
    beta = EVelocity/c
    betaDot = EAcceleration/c
    powerCGS = 2/3 * ECharge**2 / c * ( numpy.dot(betaDot, betaDot) - ( numpy.dot(numpy.cross(beta, betaDot), numpy.cross(beta, betaDot)) ))
    powerSI = powerCGS * c**2 / 10**7
    return powerSI


def CalcCyclotronFreqWithPowerLoss(PowerLoss,BField):
    Premultiplier = ECharge*BField/(EMass+EEnergy/c**2)
    return Premultiplier*(1+PowerLoss/(EMass*c**2+EEnergy))

def CalcCyclotronFreqWithPowerLossUsingEnergy(EEnergyPLoss, BField):
    return ECharge * BField / (EMass + EEnergyPLoss/c**2)

def CalcCyclotronAmpWithPowerLoss(PowerLoss,BField):
    v = (2*(EEnergy-PowerLoss)/EMass)**0.5 # velocity from K.E incorporating power loss
    return EMass * v / ECharge / BField


def CalcRelFarEField(Position,Time,EPosition,EVelocity,EAcceleration):
    # Calculates the relativistic electric field at a given position (far field)
    # for an electron located at EPosition, with acceleration and velocity given.
    Premultiplier = ECharge/(4*pi*VacPerm*c)
    r = numpy.linalg.norm(Position-EPosition) # Distance from electron (radiation point) to detector
    n = (Position-EPosition)/r # Unit vector in direction from emission point to detector position
    Premultiplier = Premultiplier * 1 / (1-numpy.dot(n,EVelocity)/c)**3
    FarFieldPart = numpy.cross( n , numpy.cross( (n-EVelocity/c) , EAcceleration/c ) ) / numpy.linalg.norm(Position-EPosition)
    # FarFieldPart = 1 / numpy.linalg.norm(Position-EPosition)
    RelFarEField = Premultiplier*FarFieldPart
    # And the time delay before it arrives at the detector will also be returned.
    TimeDelay = r/c # seconds
    return RelFarEField,TimeDelay

def CalcRelNearEField(Position,Time,EPosition,EVelocity,EAcceleration):
    # Calculates the relativistic electric field at a given position (near field)
    # for an electron located at EPosition, with acceleration and velocity given.
    Premultiplier = ECharge/(4*pi*VacPerm) 
    r = numpy.linalg.norm(Position-EPosition) # Distance from electron (radiation point) to detector
    n = (Position-EPosition)/r # Unit vector in direction from emission point to detector position
    Premultiplier = Premultiplier * 1/(1-numpy.dot(n,EVelocity)/c)**3
    
    NearFieldPart = (1-(numpy.linalg.norm(EVelocity)/c)**2)*(n-EVelocity/c) / (numpy.linalg.norm(Position-EPosition))**2
    # NearFieldPart = 0.001
    RelNearEField = Premultiplier * NearFieldPart
    TimeDelay = r/c # seconds
    return RelNearEField,TimeDelay

def CalcIncidentPowerAntenna(antennaPosArray,pointArea,Times,EPositions,EVelocities,EAccelerations):
    # Calculates the power density and power at each point in an array of points representing an antenna.
    # Also calculates the power total across all the points.
    # Uses the relativistic electric near+far field
    powerDensities = []
    powers = []
    for i in range(len(Times)): # for each timestep
        powerDensityCurrentTime = []
        powerCurrentTime = []
        for antennaPoint in antennaPosArray: # For each position on the antenna
            EField = CalcRelFarEField(antennaPoint, Times[i], EPositions[i], EVelocities[i], EAccelerations[i])[0] + CalcRelNearEField(antennaPoint, Times[i], EPositions[i], EVelocities[i], EAccelerations[i])[0]
            EFieldMagnitude = numpy.linalg.norm(EField)  
            powerDensityWorking = CalcPoyntingVectorMagnitude(EFieldMagnitude)
            powerDensityCurrentTime.append(powerDensityWorking)
            powerCurrentTime.append(powerDensityWorking*pointArea)
        powerDensities.append(powerDensityCurrentTime)
        powers.append(powerCurrentTime)
  
    powerDensities = numpy.array(powerDensities)
    powers = numpy.array(powers)
    integratedPowers = powers.sum(axis=1)
    return powerDensities,powers,integratedPowers


def HalfWaveDipoleEffectiveArea(wavelength,dipoleToEmissionAngle):
    cosFactor = ( numpy.cos(numpy.cos(dipoleToEmissionAngle)*pi/2) / numpy.sin(dipoleToEmissionAngle) )
    effectiveArea = 1.64 * cosFactor**2 * wavelength**2 / (4*pi)
    return effectiveArea

def HertzianDipoleEffectiveArea(wavelength,dipoleToEmissionAngle):
    effectiveArea = 3/(8*pi)*wavelength**2*numpy.sin(dipoleToEmissionAngle)**2
    return effectiveArea

def HalfWaveDipoleEffectiveLength(wavelength):
    return wavelength/numpy.pi




    





    
def RunSingleElectronRadiation():
    # Define the detector's position
    Resolution = 10
    x0, x1 = -0.025,0.025 # m
    y0, y1 = -0.01,0.01 # m
    z = 0.05 # m
    DetectorPositionArray = CreateDetectorArray(Resolution,x0,x1,y0,y1,z) # Detector in xy plane centred on Z axis
    
    # Calculate circular motion
    Time=0
    BField = 1 # Tesla
    AngFreq = CalcCyclotronFreqWithPowerLoss(0,BField) # 1.7563*10**11 # s^-1
    print("AngFreq=",AngFreq)
    
    # Amplitude = 0.000448 # m
    Amplitude = CalcCyclotronAmpWithPowerLoss(0,BField)
    

    # Calculate time steps
    TimeResolution = 1000
    MaxTime = 2*pi/AngFreq * 60
    TimeStep = MaxTime/TimeResolution
    TimesUsed = []

    EFieldXs = numpy.zeros(TimeResolution, dtype=object)
    EFieldYs = numpy.zeros(TimeResolution, dtype=object)
    EFieldZs = numpy.zeros(TimeResolution, dtype=object)
    EFieldMagnitudesArray = numpy.zeros(TimeResolution, dtype= object)
    PoyntingMagnitudesArray = numpy.zeros(TimeResolution, dtype=object)
    EPositionXs = numpy.zeros(TimeResolution)
    EPositionYs = numpy.zeros(TimeResolution)
    EPositionZs = numpy.zeros(TimeResolution)
    
    AngFreqArray = numpy.zeros(TimeResolution)
    AmplitudeArray = numpy.zeros(TimeResolution)
    LarmorPowerArray = numpy.zeros(TimeResolution)
    PowerLoss = 0 # Initialise total power loss, this updates as the larmor power changes in the loop
    
    for i in range(TimeResolution):
        print(i)
        # Calculate electron motion for each time step
        ZPos = 0
        
        # Update cyclotron frequency and amplitude based on total power loss
        AngFreq = CalcCyclotronFreqWithPowerLoss(PowerLoss, BField)
        Amplitude = CalcCyclotronAmpWithPowerLoss(PowerLoss, BField)
        
        EPosition, EVelocity, EAcceleration = CalcElectronCircularMotionPerp(Time,AngFreq,Amplitude,ZPos)
        
        # Calculate EField and Poynting for each time step
        #EFieldArray = CalcNonRelEFieldArray(DetectorPositionArray,Time,EPosition,EAcceleration,Resolution)
        EFieldArray = CalcNonRelEFieldArray(DetectorPositionArray,Time,EPosition,EAcceleration,Resolution) + CalcNonRelNearEFieldArray(DetectorPositionArray,Time,EPosition,EVelocity,EAcceleration,Resolution)
        EFieldMagnitudes = numpy.linalg.norm(EFieldArray,axis=(2))    
        EFieldXs[i] = EFieldArray[0:,0:,0]
        EFieldYs[i] = EFieldArray[0:,0:,1]
        EFieldZs[i] = EFieldArray[0:,0:,2]
        EFieldMagnitudesArray[i] = EFieldMagnitudes
        PoyntingMagnitudesArray[i] = CalcPoyntingVectorMagnitude(EFieldMagnitudes)
        
        EPositionXs[i] = EPosition[0]
        EPositionYs[i] = EPosition[1]
        EPositionZs[i] = EPosition[2]
        
        TimesUsed.append(Time)
        Time = Time + TimeStep
        

        LarmorPower = CalcLarmorPower(EAcceleration)
        PowerLoss += LarmorPower * TimeStep
        
        LarmorPowerArray[i] = LarmorPower
        AngFreqArray[i] = AngFreq
        AmplitudeArray[i] = Amplitude
        
    # end loop
    
    # Calculate Power from Poynting vector
    # PixelArea = (x1-x0)/Resolution*(y1-y0)/Resolution # Instead use power/m^2
    PowerArray = PoyntingMagnitudesArray # *PixelArea
        
    # Set colour bar graph limits
    ColourMinEMag,ColourMaxEMag = FindColourBounds(EFieldMagnitudesArray,TimeResolution)
    ColourMinPoynting,ColourMaxPoynting = FindColourBounds(PoyntingMagnitudesArray,TimeResolution)
    ColourMinPowerArray,ColourMaxPowerArray = FindColourBounds(PowerArray,TimeResolution)
    ColourMinEFXs,ColourMaxEFXs = FindColourBounds(EFieldXs,TimeResolution)
    ColourMinEFYs,ColourMaxEFYs = FindColourBounds(EFieldYs,TimeResolution)
    ColourMinEFZs,ColourMaxEFZs = FindColourBounds(EFieldZs,TimeResolution)
        
    xs = DetectorPositionArray[0:,0:,0]
    ys = DetectorPositionArray[0:,0:,1]

    if TimeResolution < 50:
        for i in range(TimeResolution):
            # Plot and output images
            figsize = [7, 8]     # figure size, inches
            fig1, ax = pyplot.subplots(nrows=2, ncols=1, figsize=figsize)
            fig1graph0 = ax[0].imshow(EFieldMagnitudesArray[i]*10**6, extent=(x0,x1,y0,y1))
            fig1graph1 = ax[1].imshow(PowerArray[i], extent=(x0,x1,y0,y1))
            pyplot.xlabel("x (m)")
            pyplot.ylabel("y (m)")
            colourbar = pyplot.colorbar(fig1graph0, ax=ax[0], orientation="vertical")
            colourbar.set_label(r"Magnitude ($\mu$V/m)")
            colourbar = pyplot.colorbar(fig1graph1, ax=ax[1], orientation="vertical")
            colourbar.set_label(r"Power Density (W/m$^2$)")
            fig1graph0.set_clim(ColourMinEMag*10**6,ColourMaxEMag*10**6)
            fig1graph1.set_clim(ColourMinPowerArray,ColourMaxPowerArray)
            ax[0].set_title("E Field Magnitude, t=%ss" % '%.2g' % TimesUsed[i])
            ax[1].set_title("Power")
            pyplot.tight_layout()
            fig1.savefig("Images\Poynting\Poynting%s" % i)
            
            fig2,ax2 = pyplot.subplots(nrows=1, ncols=1, figsize=[8,5])
            fig2graph0 = ax2.quiver(xs,ys,EFieldXs[i],EFieldYs[i])
            ax2.set_title("E Field XY Components, t=%ss" % '%.2g' % TimesUsed[i])
            pyplot.xlabel("x (m)")
            pyplot.ylabel("y (m)")
            pyplot.tight_layout()
            fig2.savefig("Images\Polarity\EFieldPolarity%s" % i)
    
            pyplot.close(fig1) 
            pyplot.close(fig2)
            
            fig4,ax4 = pyplot.subplots(nrows=3, ncols=1, figsize=[8,12])
            fig4graph0 = ax4[0].imshow(EFieldXs[i]*10**6, extent=(x0,x1,y0,y1))
            fig4graph1 = ax4[1].imshow(EFieldYs[i]*10**6, extent=(x0,x1,y0,y1))
            fig4graph2 = ax4[2].imshow(EFieldZs[i]*10**6, extent=(x0,x1,y0,y1))
            colourbar = pyplot.colorbar(fig4graph0, ax=ax4[0], orientation="vertical")
            colourbar.set_label(r"Magnitude ($\mu$V/m)")
            colourbar = pyplot.colorbar(fig4graph1, ax=ax4[1], orientation="vertical")
            colourbar.set_label(r"Magnitude ($\mu$V/m)")
            colourbar = pyplot.colorbar(fig4graph2, ax=ax4[2], orientation="vertical")
            colourbar.set_label(r"Magnitude ($\mu$V/m)")
            fig4graph0.set_clim(ColourMinEFXs*10**6,ColourMaxEFXs*10**6)
            fig4graph1.set_clim(ColourMinEFYs*10**6,ColourMaxEFYs*10**6)
            fig4graph2.set_clim(ColourMinEFZs*10**6,ColourMaxEFZs*10**6)
            ax4[0].set_title("E Field x Component, t=%ss" % '%.2g' %TimesUsed[i])
            ax4[1].set_title("E Field y Component")
            ax4[2].set_title("E Field z Component")
            pyplot.xlabel("x (m)")
            pyplot.ylabel("y (m)")
            pyplot.tight_layout()
            fig4.savefig("Images\EFieldComponents\EFieldComponents%s" % i)
            pyplot.close(fig4)
            
        # end loop

    fig3,ax3 = pyplot.subplots(nrows=3, ncols=1, figsize=[10,15])
    fig3graph0 = ax3[0].plot(TimesUsed,EPositionXs)
    fig3graph1 = ax3[1].plot(TimesUsed,EPositionYs)   
    fig3graph2 = ax3[2].plot(TimesUsed,EPositionZs)
    ax3[0].set_title("Electron Position")
    ax3[0].set_xlabel('t (s)')
    ax3[0].set_ylabel('x (m)')
    ax3[1].set_xlabel('t (s)')
    ax3[1].set_ylabel('y (m)')
    ax3[2].set_xlabel('t (s)')
    ax3[2].set_ylabel('z (m)')
    fig3.savefig("Images\Position\EPosition")
    
    figLarm,axLarm = pyplot.subplots(nrows=1, ncols=1, figsize=[11,7])
    axLarm.plot(TimesUsed,LarmorPowerArray)
    axLarm.set_title("Larmor Power")
    axLarm.set_xlabel('t (s)')
    axLarm.set_ylabel('Larmor Power (W)')
    figLarm.savefig("Images\Misc\LarmorPower")
    
    figAngFreq,axAngFreq = pyplot.subplots(nrows=1, ncols=1, figsize=[11,10])
    axAngFreq.plot(TimesUsed,AngFreqArray)
    axAngFreq.set_title("Angular Frequency")
    axAngFreq.set_xlabel('t (s)')
    axAngFreq.set_ylabel('Angular Frequency (Hz)')
    figAngFreq.savefig("Images\Misc\AngFreq")

    figAmplitude,axAmplitude = pyplot.subplots(nrows=1, ncols=1, figsize=[11,10])
    axAmplitude.plot(TimesUsed,AmplitudeArray)
    axAmplitude.set_title("Cyclotron Radius")
    axAmplitude.set_xlabel('t (s)')
    axAmplitude.set_ylabel('Cyclotron Radius (m)')
    figAmplitude.savefig("Images\Misc\CycloRadius")

    pyplot.close(fig3)
    pyplot.close(figLarm)
    pyplot.close(figAngFreq)
    pyplot.close(figAmplitude)
    
    figCycloFreq,axCycloFreq = pyplot.subplots(nrows=1, ncols=1, figsize = [11,7])
    axCycloFreq.plot(TimesUsed,AngFreqArray/(2*pi))
    axCycloFreq.set_title("Cyclotron Frequency")
    axCycloFreq.set_xlabel("t (s)")
    axCycloFreq.set_ylabel("Cyclotron Frequency (Hz)")
    figCycloFreq.tight_layout()
    figCycloFreq.savefig("Images\Misc\CycloFreq")
    pyplot.close(figCycloFreq)

# Run the code
if __name__ == "__main__":
    RunSingleElectronRadiation()
    

    
import numpy
import matplotlib.pyplot as pyplot

def CalculateGamma(kineticEnergy):
    return kineticEnergy/(eMass*c**2)+1

def CalcVelocity(gamma):
    return numpy.sqrt( c**2 * (1-1/gamma**2) )

def LarmorPower(acceleration):
    return eCharge**2 * acceleration**2 / (6 * pi * vacPermit * c**3)

def AngularFrequency(kineticEnergyJ, B):
    return eCharge * B / (eMass + kineticEnergyJ/c**2)

def CalcAcceleration(angularFrequency, cyclotronRadius):
    a = angularFrequency**2 * cyclotronRadius
    return a
    
def CyclotronRadius(perpendicularVelocity, B, gamma, angularFrequency):
    r = (eCharge*perpendicularVelocity*B) / (gamma * eMass * angularFrequency**2)
    return r

eMass = 9.10938356e-31
eCharge = 1.602176634e-19
c = 299792458
pi = numpy.pi
KEeV = 18600
KEJ = KEeV*eCharge
vacPermit = 8.8541878128*10**-12

if __name__ == "__main__":
    gamma = KEJ/(eMass*c**2)+1
    print("Gamma:",gamma)
    vsquared = c**2 * (1-1/gamma**2)
    v = numpy.sqrt(vsquared)
    
    print("Velocity:",v)
    B= 1.0
    angularFrequency = eCharge*B/(gamma*eMass)
    
    print("Angular Frequency:",angularFrequency)
    print("Frequency:",angularFrequency/(2*numpy.pi))
    
    # Axial oscillation frequency calculator
    
    pitchAngleDeg = 90
    trapLength = 1 # metres
    
    axialVelocity = v * numpy.cos(pitchAngleDeg/180*numpy.pi)
    print("Axial Velocity:",axialVelocity)
    axialFrequency = axialVelocity/trapLength
    print("Axial Frequency:",axialFrequency)
    
    perpendicularVelocity = numpy.sqrt(v**2 - axialVelocity**2)
    
    
    r = CyclotronRadius(perpendicularVelocity, B, gamma, angularFrequency)
    print("Cyclotron Radius:",r)
    
    
    electronKE = KEJ
    timeStep = 1e-10
    times = numpy.arange(0, 1e-9, step = timeStep)
    powers = []
    electronKEs = []
    for t in range(len(times)):
        # At each time, we want to know the power output
        # This requires knowing the acceleration
        # Which requires knowing the cyclotron radius and angular frequency at each time step
        # This needs gamma, which depends on the electron's energy
        #print("Electron KE:",electronKE)
        gamma = electronKE/(eMass*c**2)+1
        #print("Gamma:",gamma)
        angularFrequency = eCharge*1/(gamma*eMass)
        #print("AngFreq:",angularFrequency)
        
        vsquared = c**2 * (1-1/gamma**2)
        v = numpy.sqrt(vsquared)
        axialVelocity = v * numpy.cos(pitchAngleDeg/180*numpy.pi)
        perpendicularVelocity = numpy.sqrt(v**2 - axialVelocity**2)
        
        
        r = (eCharge*perpendicularVelocity*B) / (gamma * eMass * angularFrequency**2)
        #print("r:",r)
        a = CalcAcceleration(angularFrequency, r)
        #print("a",a)
        currentPower = LarmorPower(a)
        print("Power:",currentPower)
        electronKEs.append(electronKE)   
        
        electronKE -= currentPower * (timeStep)
    
        powers.append(currentPower)
        
    electronKEs = numpy.array(electronKEs)
    
    
    figPowerOverTime,axPowerOverTime = pyplot.subplots(1,1,figsize = [10,8])
    axPowerOverTime.plot(times,powers)
    axPowerOverTime.set_xlabel("Time (s)")
    axPowerOverTime.set_ylabel("Power (W)")
    
    figEnergy, axEnergy = pyplot.subplots(1,1,figsize=[10,8])
    axEnergy.plot(times,electronKEs/eCharge)
    axEnergy.set_xlabel("Time (s)")
    axEnergy.set_ylabel("Energy (eV)")
        
    initialKineticEnergy = KEJ
    finalKineticEnergy = electronKE
    
    print("Initial KE (eV):", initialKineticEnergy/eCharge)
    print("Final KE (eV):", finalKineticEnergy/eCharge)
    
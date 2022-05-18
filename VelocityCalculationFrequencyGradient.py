import numpy

eMass = 9.10938356e-31
eCharge = 1.602176634e-19
c = 299792458

def CalculateFrequencyGradient(initialKEeV, finalKEeV, totalTime, B):
    initialKEJ = initialKEeV*eCharge
    finalKEJ = finalKEeV*eCharge
    initialGamma = initialKEJ/(eMass*c**2)+1
    initialAngularFrequency = eCharge*B/(initialGamma*eMass)
    initialFrequency = initialAngularFrequency/(2*numpy.pi)
    finalGamma = finalKEJ/(eMass*c**2)+1
    finalAngularFrequency = eCharge*B/(finalGamma*eMass)
    finalFrequency = finalAngularFrequency/(2*numpy.pi) 
    
    frequencyGradient = (finalFrequency-initialFrequency)/totalTime
    return frequencyGradient 


def CalculateFrequency(initialKEeV, B):
    initialKEJ = initialKEeV*eCharge
    initialGamma = initialKEJ/(eMass*c**2)+1
    initialAngularFrequency = eCharge*B/(initialGamma*eMass)
    initialFrequency = initialAngularFrequency/(2*numpy.pi)
    return initialFrequency




if __name__ == "__main__":
    KEeV = 18600
    KEJ = KEeV*eCharge
    
    B=1.00000
    
    gamma = KEJ/(eMass*c**2)+1
    print("Gamma:",gamma)
    vsquared = c**2 * (1-1/gamma**2)
    v = numpy.sqrt(vsquared)
    
    print("Velocity:",v)
    
    angularFrequency = eCharge*B/(gamma*eMass)
    
    print("Angular Frequency:",angularFrequency)
    print("Frequency:",angularFrequency/(2*numpy.pi))
    
    
    # Axial oscillation frequency calculator
    pitchAngleDeg = 90
    trapLength = 0.08*4 # metres
    
    axialVelocity = v * numpy.cos(pitchAngleDeg/180*numpy.pi)
    print("Axial Velocity:",axialVelocity)
    axialFrequency = axialVelocity/trapLength
    print("Axial Frequency:",axialFrequency)
    
    perpendicularVelocity = numpy.sqrt(v**2 - axialVelocity**2)
    
    
    r = (eCharge*perpendicularVelocity*B) / (gamma * eMass * angularFrequency**2) 
    print("Cyclotron Radius:",r)



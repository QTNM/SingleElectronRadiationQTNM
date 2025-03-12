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
protonMass = 1.67262192595e-27
neutronMass = 1.67492750056e-27 # kg
eCharge = 1.602176634e-19
c = 299792458
pi = numpy.pi
KEeV = 18566.66
KEJ = KEeV*eCharge
vacPermit = 8.8541878128*10**-12

if __name__ == "__main__":
    gamma = KEJ/(eMass*c**2)+1
    print("Gamma:",gamma)
    vsquared = c**2 * (1-1/gamma**2)
    v = numpy.sqrt(vsquared)
    
    print("Velocity:",v)
    #B= 0.7000#2313059249833
    B = 1.0
    angularFrequency = eCharge*B/(gamma*eMass)
    
    print("Angular Frequency:",angularFrequency)
    print("Frequency:",angularFrequency/(2*numpy.pi))
    print("Wavelength:",c/(angularFrequency/(2*numpy.pi)), "m")
    
    # Axial oscillation frequency calculator
    
    pitchAngleDeg = 90
    trapLength = 0.2 # metres
    
    axialVelocity = v * numpy.cos(pitchAngleDeg/180*numpy.pi)
    print("Axial Velocity:",axialVelocity)
    axialFrequency = axialVelocity/trapLength
    print("Axial Frequency:",axialFrequency)
    
    perpendicularVelocity = numpy.sqrt(v**2 - axialVelocity**2)
    
    
    r = CyclotronRadius(perpendicularVelocity, B, gamma, angularFrequency)
    print("Cyclotron Radius:",r)
    
    
    
    print("######### Recoil Energy Calculation #########")
    electronVelocity = v # implicit assumption that the KE is equal to the endpoint energy...
    electronMomentum = electronVelocity * gamma * eMass
    endpointEnergy = 18566.66 # eV
    # Assuming 0 neutrino velocity:
    eMass_eV = eMass * c * c / eCharge
    daughterMass_eV = (2*protonMass + neutronMass) * c * c / eCharge
    daughterMass = 2*protonMass + neutronMass
    electronKEeV = endpointEnergy - eMass_eV - daughterMass_eV # 
    daughterMomentum = -electronMomentum
    daughterVelocity = daughterMomentum/daughterMass
    
    Q = endpointEnergy * eCharge
    print("Q",Q)
    E_md = daughterMass * c**2
    
    daughterMassEnergy = daughterMass * c**2
    electronMassEnergy = eMass*c**2
    print("daughter mass energy:",daughterMassEnergy)
    print("electron mass energy:",electronMassEnergy)
    
    E_daughter = (Q**2 - electronMassEnergy**2 + daughterMassEnergy**2) / (2*Q)
    print("E_daughter:",E_daughter)
    E_rec = (Q**2 + 2*Q*electronMassEnergy) / (2 * (daughterMassEnergy + Q + electronMassEnergy) )
    
    E_rec_eV = E_rec / eCharge
    print("Electron momentum:",electronMomentum / eCharge)
    print("Recoil energy:",E_rec_eV)
    
    
    
    
    
    
    
    
    electronKE = KEJ
    timeStep = 1e-11
    times = numpy.arange(0, 1e-10, step = timeStep)
    powers = []
    electronKEs = []
    frequencies = []
    radii = []
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
        radii.append(r)
        #print("r:",r)
        a = CalcAcceleration(angularFrequency, r)
        if t==0:
            print("Initial acceleration:",a)
        #print("a",a)
        currentPower = LarmorPower(a)
        # print("Power:",currentPower)
        electronKEs.append(electronKE)   
        
        frequencies.append(AngularFrequency(electronKE, B) / (2*numpy.pi))
        
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
    
    figFrequency, axFrequency = pyplot.subplots(1,1,figsize=[10,8])
    axFrequency.plot(times,frequencies)
    axFrequency.set_xlabel("Time (s)")
    axFrequency.set_ylabel("Frequency (Hz)")
    
    print("Frequency Gradients:", numpy.gradient(frequencies, timeStep))
        
    initialKineticEnergy = KEJ
    finalKineticEnergy = electronKEs[-1]
    
    print("Initial KE (eV):", initialKineticEnergy/eCharge)
    print("Final KE (eV):", finalKineticEnergy/eCharge)
    
    
    figRadii, axRadii = pyplot.subplots(1, 1, figsize=[16,8])
    axRadii.plot(times, radii)
    
    
    print("Test numbers:")
    testE1 = 18.575e3
    testE2 = 18.4e3
    testFreq1 = AngularFrequency(testE1*eCharge, B=0.7) / (2*numpy.pi)
    testFreq2 = AngularFrequency(testE2*eCharge, B=0.7) / (2*numpy.pi)
    print("testFreq1:", testFreq1)
    print("testFreq2:", testFreq2)
    print("Difference:", testFreq2 - testFreq1)
    print("Gradient:", (testFreq2 - testFreq1) / 1e-4)
    
    
    
    
    

    
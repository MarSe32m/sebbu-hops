//
//  main.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import PythonKit
import PythonKitUtilities

#if os(macOS)
PythonLibrary.useLibrary(at: "/Library/Frameworks/Python.framework/Versions/3.12/Python")
#elseif os(Linux)
//PythonLibrary.useLibrary(at: "/usr/lib64/libpython3.11.so.1.0")
PythonLibrary.useLibrary(at: "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0")
#elseif os(Windows)
//TODO: Set the library path on Windows machine
#endif



IBMFockStateAmplitudesExample(endTime: 1000)
testBCFFittings()
//ornsteinUhlenbeckExample2()
ornsteinUhlenbeckExample3()
drivenDissipativeCavityMode(endTime: 50, omegaX: 2.5, g: 0.05, omegaC: 2.5, gammaMinus: 0.2, gammaPlus: 0.0)
//radiativeDampingExample(realizations: 100, endTime: 100)
IBMExampleUnified(realizations: 10000, endTime: 7)
IBMExample(realizations: 10000, endTime: 7, plotBCF: false)



//ornsteinUhlenbeckExample()
//basisTest()
HOPSvsNMQSD(realizations: 1)
OperatorNMQSDvsHOPS(realizations: 1, endTime: 10)
let (heomTSpace, heomX, heomY, heomZ) = IBMExampleHEOM(endTime: 750.0, depth: 5, plotBCF: false)
let (hopsTSpace, hopsX, hopsY, hopsZ) = _linearIBM(endTime: 750.0, realizations: 8192, plotBCF: false)
let (exactTSpace, exactX, exactY, exactZ) = _ibmExact(endTime: 750.0)
let (nmqsdTSpace, nmqsdX, nmqsdY, nmqsdZ) = _linearNMQSDIBM(endTime: 750.0, realizations: 8192, plotBCF: false)

plt.figure()
plt.plot(x: hopsTSpace, y: hopsX, label: "HOPS X")
plt.plot(x: hopsTSpace, y: hopsY, label: "HOPS Y")
plt.plot(x: hopsTSpace, y: hopsZ, label: "HOPS Z")

plt.plot(x: nmqsdTSpace, y: nmqsdX, label: "NMQSD X")
plt.plot(x: nmqsdTSpace, y: nmqsdY, label: "NMQSD Y")
plt.plot(x: nmqsdTSpace, y: nmqsdZ, label: "NMQSD Z")

plt.plot(x: heomTSpace, y: heomX, label: "MEH X")
plt.plot(x: heomTSpace, y: heomY, label: "MEH Y")
plt.plot(x: heomTSpace, y: heomZ, label: "MEH Z")


plt.plot(x: exactTSpace, y: exactX, label: "Exact X", linestyle: "--")
plt.plot(x: exactTSpace, y: exactY, label: "Exact Y", linestyle: "--")
plt.plot(x: exactTSpace, y: exactZ, label: "Exact Z", linestyle: "--")

plt.legend()
plt.xlabel("t")
plt.ylabel("<O>")
plt.show()
plt.close()

testRFSpectrum(omegaX: 1350.0, detuning: 0.0, omegaC: 1350.0, rabi: 0.5, a: 0.5, ksi: 1.447, kappa: 0.0175, gammaR: 0.175, temperature: 0.0, steadyStateTime: 200, endTime: 500, trajectories: 8192 << 4)
//testSingleParticleLinearHOPSTwoTimeCorrelationFunction(trajectories: 128)
//testSingleParticleNonLinearHOPSTwoTimeCorrelationFunction(trajectories: 128)
//radiativeDampingPlusPumpingMultiParticleExample(realizations: 8192, endTime: 750.0)
//radiativeDampingPlusPumpingExample(realizations: 2048, endTime: 750.0)
//radiativeDampingExample(realizations: 1<<10, endTime: 750.0)
//testHOPSDecomposedPropagation(endTime: 120.0, at: 3.5)
//QFunctionPlot(endTime: 10.0)
//HOPSvsNMQSD(realizations: 1, endTime: 1750.0)
//noiseGenerationExample()
//multiNoiseGenerationExample()
 

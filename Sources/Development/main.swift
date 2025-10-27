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
testRFSpectrum(omegaX: 1350.0, detuning: 0.0, omegaC: 1350.0, rabi: 0.175, a: 0.27, ksi: 1.447, kappa: 0.0175, gammaR: 0.0175, temperature: 0.0, steadyStateTime: 750, endTime: 1600, trajectories: 1024 << 7)
//testSingleParticleLinearHOPSTwoTimeCorrelationFunction(trajectories: 128)
//testSingleParticleNonLinearHOPSTwoTimeCorrelationFunction(trajectories: 128)
//radiativeDampingPlusPumpingMultiParticleExample(realizations: 8192, endTime: 750.0)
//radiativeDampingPlusPumpingExample(realizations: 2048, endTime: 750.0)
radiativeDampingExample(realizations: 1<<10, endTime: 750.0)
//testHOPSDecomposedPropagation(endTime: 120.0, at: 3.5)
//QFunctionPlot(endTime: 10.0)
//HOPSvsNMQSD(realizations: 1, endTime: 1750.0)
IBMExample(realizations: 8196 * 2, endTime: 10.0, plotBCF: true)
//noiseGenerationExample()
//multiNoiseGenerationExample()

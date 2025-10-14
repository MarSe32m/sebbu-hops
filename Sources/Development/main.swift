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

testHOPSDecomposedPropagation(endTime: 120.0, at: 3.5)
QFunctionPlot(endTime: 10.0)
HOPSvsNMQSD(realizations: 1, endTime: 1750.0)
IBMExample(realizations: 8196 * 2, endTime: 10.0, plotBCF: true)
noiseGenerationExample()

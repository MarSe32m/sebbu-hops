//
//  IBM.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import HOPS
import SebbuScience
import PythonKitUtilities

public func HOPSvsNMQSD(realizations: Int, endTime: Double = 7.0) {
    let A = 0.027
    let omegaC = 1.447
    
    let renormalizationEnergy = Quad.integrate(a: 0, b: .infinity) { omega in
        spectralDensityByOmega(omega: omega, A: A, omegaC: omegaC)
    }
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 500)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    print(G)
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 3)
    let nmqsdCalculation = NMQSDCalculation(dimension: 2, L: L, G: G, W: W)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(renormalizationEnergy)], rows: 2, columns: 2)
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }

    let noiseGenerationStart = ContinuousClock().now
    let noises = zGenerator.generateParallel(count: realizations)
    let noiseGenerationEnd = ContinuousClock().now
    print("Noise generation time: \(noiseGenerationEnd - noiseGenerationStart)")
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    var linearStart = ContinuousClock().now
    let linearTrajectories = noises.parallelMap { z in
        hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z, stepSize: 0.01)
    }
    var linearEnd = ContinuousClock().now
    print("Linear HOPS time:      \(linearEnd - linearStart)")
    
    linearStart = .now
    let linearNMQSDTrajectories = noises.parallelMap { z in 
        var _L: Matrix<Complex<Double>> = .zeros(rows: L.rows, columns: L.columns)
        return nmqsdCalculation.solveLinear(end: endTime, initialState: initialState, H: H, z: z) { t, z in 
            var bcfIntegral: Complex<Double> = .zero
            for i in G.indices {
                bcfIntegral += G[i] / W[i] * (.one - .exp(-t * W[i]))
            }
            _L.copyElements(from: L)
            _L.multiply(by: bcfIntegral)
            return _L
        }
    }
    linearEnd = .now
    print("Linear NMQSD time:     \(linearEnd - linearStart)")

    var nonLinearStart = ContinuousClock().now
    let nonLinearTrajectories = noises.parallelMap { z in
        hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, stepSize: 0.01)
    }
    var nonLinearEnd = ContinuousClock().now
    print("Non-linear HOPS time:  \(nonLinearEnd - nonLinearStart)")
    
    let nonLinearShiftedTrajectories = noises.parallelMap { z in
        hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, shiftType: .meanField, stepSize: 0.01)
    }
    
    nonLinearStart = .now
    let nonLinearNMQSDTrajectories = noises.parallelMap { z in 
        var _L: Matrix<Complex<Double>> = .zeros(rows: L.rows, columns: L.columns)
        return nmqsdCalculation.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z) { t, z in 
            var bcfIntegral: Complex<Double> = .zero
            for i in G.indices {
                bcfIntegral += G[i] / W[i] * (.one - .exp(-t * W[i]))
            }
            _L.copyElements(from: L)
            _L.multiply(by: bcfIntegral)
            return _L
        }
    }
    nonLinearEnd = .now
    print("Non-linear NMQSD time: \(nonLinearEnd - nonLinearStart)")

    let linearTSpace = linearTrajectories[0].tSpace
    var linearRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory) in linearTrajectories {
        let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory)
        if linearRho.isEmpty {
            linearRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                linearRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }

    let linearNMQSDTSpace = linearNMQSDTrajectories[0].tSpace
    var linearNMQSDRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory) in linearNMQSDTrajectories {
        let _rho = nmqsdCalculation.mapLinearToDensityMatrix(trajectory)
        if linearNMQSDRho.isEmpty {
            linearNMQSDRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                linearNMQSDRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }
    
    let nonLinearTSpace = nonLinearTrajectories[0].tSpace
    var nonLinearRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory, _) in nonLinearTrajectories {
        let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory, normalize: true)
        if nonLinearRho.isEmpty {
            nonLinearRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                nonLinearRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }
    
    let nonLinearShiftedTSpace = nonLinearShiftedTrajectories[0].tSpace
    var nonLinearShiftedRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory, _) in nonLinearShiftedTrajectories {
        let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory, normalize: true)
        if nonLinearShiftedRho.isEmpty {
            nonLinearShiftedRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                nonLinearShiftedRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }
    
    let nonLinearNMQSDTSpace = nonLinearNMQSDTrajectories[0].tSpace
    var nonLinearNMQSDRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory) in nonLinearNMQSDTrajectories {
        let _rho = nmqsdCalculation.mapNonLinearToDensityMatrix(trajectory)
        if nonLinearNMQSDRho.isEmpty {
            nonLinearNMQSDRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                nonLinearNMQSDRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }

    let linearHOPSZ = linearRho.map { $0[0, 0].real - $0[1, 1].real }
    let linearHOPSX = linearRho.map { 2 * $0[0, 1].real }
    let linearHOPSY = linearRho.map { 2 * $0[0, 1].imaginary }
    
    let nonLinearHOPSZ = nonLinearRho.map { $0[0, 0].real - $0[1, 1].real }
    let nonLinearHOPSX = nonLinearRho.map { 2 * $0[0, 1].real }
    let nonLinearHOPSY = nonLinearRho.map { 2 * $0[0, 1].imaginary }

    let nonLinearShiftedHOPSZ = nonLinearShiftedRho.map { $0[0, 0].real - $0[1, 1].real }
    let nonLinearShiftedHOPSX = nonLinearShiftedRho.map { 2 * $0[0, 1].real }
    let nonLinearShiftedHOPSY = nonLinearShiftedRho.map { 2 * $0[0, 1].imaginary }
    
    let linearNMQSDZ = linearNMQSDRho.map { $0[0, 0].real - $0[1, 1].real }
    let linearNMQSDX = linearNMQSDRho.map { 2 * $0[0, 1].real }
    let linearNMQSDY = linearNMQSDRho.map { 2 * $0[0, 1].imaginary }
    
    let nonLinearNMQSDZ = nonLinearNMQSDRho.map { $0[0, 0].real - $0[1, 1].real }
    let nonLinearNMQSDX = nonLinearNMQSDRho.map { 2 * $0[0, 1].real }
    let nonLinearNMQSDY = nonLinearNMQSDRho.map { 2 * $0[0, 1].imaginary }
    
    print(linearNMQSDTSpace.count, linearNMQSDX.count)
    print(linearNMQSDTSpace.count, linearNMQSDY.count)
    print(linearNMQSDTSpace.count, linearNMQSDZ.count)
    // Compare linear trajectories
    plt.figure()
    plt.plot(x: linearNMQSDTSpace, y: linearNMQSDX, label: "NMQSD X")
    plt.plot(x: linearNMQSDTSpace, y: linearNMQSDY, label: "NMQSD Y")
    plt.plot(x: linearNMQSDTSpace, y: linearNMQSDZ, label: "NMQSD Z")
    plt.plot(x: linearTSpace, y: linearHOPSX, label: "HOPS X", linestyle: "--")
    plt.plot(x: linearTSpace, y: linearHOPSY, label: "HOPS Y", linestyle: "--")
    plt.plot(x: linearTSpace, y: linearHOPSZ, label: "HOPS Z", linestyle: "--")
    plt.title("Linear HOPS vs NMQSD")
    plt.legend()
    plt.xlabel("t")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x: nonLinearNMQSDTSpace, y: nonLinearNMQSDX, label: "NMQSD X")
    plt.plot(x: nonLinearNMQSDTSpace, y: nonLinearNMQSDY, label: "NMQSD Y")
    plt.plot(x: nonLinearNMQSDTSpace, y: nonLinearNMQSDZ, label: "NMQSD Z")
    plt.plot(x: nonLinearTSpace, y: nonLinearHOPSX, label: "HOPS X", linestyle: "--")
    plt.plot(x: nonLinearTSpace, y: nonLinearHOPSY, label: "HOPS Y", linestyle: "--")
    plt.plot(x: nonLinearTSpace, y: nonLinearHOPSZ, label: "HOPS Z", linestyle: "--")
    plt.title("Non-linear HOPS vs NMQSD")
    plt.legend()
    plt.xlabel("t")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(x: nonLinearNMQSDTSpace, y: nonLinearNMQSDX, label: "NMQSD X")
    plt.plot(x: nonLinearNMQSDTSpace, y: nonLinearNMQSDY, label: "NMQSD Y")
    plt.plot(x: nonLinearNMQSDTSpace, y: nonLinearNMQSDZ, label: "NMQSD Z")
    plt.plot(x: nonLinearShiftedTSpace, y: nonLinearShiftedHOPSX, label: "HOPS X", linestyle: "--")
    plt.plot(x: nonLinearShiftedTSpace, y: nonLinearShiftedHOPSY, label: "HOPS Y", linestyle: "--")
    plt.plot(x: nonLinearShiftedTSpace, y: nonLinearShiftedHOPSZ, label: "HOPS Z", linestyle: "--")
    plt.title("Non-linear shifted HOPS vs NMQSD")
    plt.legend()
    plt.xlabel("t")
    plt.show()
    plt.close()
}

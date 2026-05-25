//
//  NMQSDOperatorFormalismTest.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 3.5.2026.
//

import HOPS
import SebbuScience
import PythonKitUtilities

public func OperatorNMQSDvsHOPS(realizations: Int, endTime: Double = 7.0) {
    let A = 0.87
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
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 5)
    let hierarchyForShifted = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 4)
    let nmqsdCalculation = NMQSDCalculation(dimension: 2, L: L, G: G, W: W)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
//    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
//        spectralDensity(omega: omega, A: A, omegaC: omegaC)
//    }
    let zGenerator = ZeroNoiseProcessGenerator()

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
        nmqsdCalculation.solveLinear2(end: endTime, initialState: initialState, H: H, z: z, includePropagator: true, stepSize: 0.01)
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
        hierarchyForShifted.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, shiftType: .meanField, stepSize: 0.01)
    }
    
    nonLinearStart = .now
    let nonLinearNMQSDTrajectories = noises.parallelMap { z in
        nmqsdCalculation.solveNonLinear2(end: endTime, initialState: initialState, H: H, z: z, stepSize: 0.01)
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
    for (_, trajectory, _) in linearNMQSDTrajectories {
        let _rho = nmqsdCalculation.mapTrajectoryToDensityMatrix(trajectory)
        if linearNMQSDRho.isEmpty {
            linearNMQSDRho = _rho.map { $0 / Double(realizations) }
        } else {
            for i in 0..<_rho.count {
                linearNMQSDRho[i].add(_rho[i], multiplied: 1 / Double(realizations))
            }
        }
    }
    if !linearNMQSDTrajectories[0].propagator.isEmpty {
        let propagator = linearNMQSDTrajectories[0].propagator
        let X: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .one, .zero], rows: 2, columns: 2)
        let Y: Matrix<Complex<Double>> = .init(elements: [.zero, -.i, .i, .zero], rows: 2, columns: 2)
        let Z: Matrix<Complex<Double>> = .init(elements: [.one, .zero, .zero, -.one], rows: 2, columns: 2)
        let lambda = propagator.map { 0.5 * $0.trace }
        let lambdaX = propagator.map { 0.5 * $0.dot(X).trace }
        let lambdaY = propagator.map { 0.5 * $0.dot(Y).trace }
        let lambdaZ = propagator.map { 0.5 * $0.dot(Z).trace }
        let z: [Complex<Double>] = linearNMQSDTSpace.map { noises[0].sample($0) }
        plt.figure()
        plt.plot(x: linearNMQSDTSpace, y: lambdaX.real, label: "Re X")
        plt.plot(x: linearNMQSDTSpace, y: lambdaX.imaginary, label: "Im X")
        plt.plot(x: linearNMQSDTSpace, y: lambdaY.real, label: "Re Y", linestyle: "--")
        plt.plot(x: linearNMQSDTSpace, y: lambdaY.imaginary, label: "Im Y", linestyle: "--")
        plt.plot(x: linearNMQSDTSpace, y: z.real, label: "Re z")
        plt.plot(x: linearNMQSDTSpace, y: z.imaginary, label: "Im z")
        plt.legend()
        plt.show()
        plt.close()
        
        plt.figure()
        
//        plt.plot(x: linearNMQSDTSpace, y: lambda.real, label: "Re 1")
//        plt.plot(x: linearNMQSDTSpace, y: lambda.imaginary, label: "Im 1")
//        plt.plot(x: linearNMQSDTSpace, y: lambdaZ.real, label: "Re Z", linestyle: "--")
//        plt.plot(x: linearNMQSDTSpace, y: lambdaZ.imaginary, label: "Im Z", linestyle: "--")
        //plt.plot(x: linearNMQSDTSpace, y: zip(lambda, lambdaZ).map { $0 + $1 }.real, label: "Re (l + z)")
        plt.plot(x: linearNMQSDTSpace, y: zip(lambda, lambdaZ).map { $0 - $1 }.imaginary, label: "Im (l + z)")
        plt.plot(x: linearNMQSDTSpace, y: zip(lambda, lambdaZ).map { $0 + $1 }.real, label: "Re (l - z)", linestyle: "-.")
        plt.plot(x: linearNMQSDTSpace, y: zip(lambda, lambdaZ).map { $0 - $1 }.imaginary, label: "Im (l - z)", linestyle: "--")
//        plt.plot(x: linearNMQSDTSpace, y: z.real, label: "Re z")
//        plt.plot(x: linearNMQSDTSpace, y: z.imaginary, label: "Im z")
        plt.legend()
        plt.show()
        plt.close()
    }
    
    
    let nonLinearTSpace = nonLinearTrajectories[0].tSpace
    var nonLinearRho: [Matrix<Complex<Double>>] = []
    for (_, trajectory) in nonLinearTrajectories {
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
    for (_, trajectory) in nonLinearShiftedTrajectories {
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
    for (_, trajectory, _) in nonLinearNMQSDTrajectories {
        let _rho = nmqsdCalculation.mapTrajectoryToDensityMatrix(trajectory, normalize: true)
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

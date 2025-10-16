//
//  RadiativeDamping.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 15.10.2025.
//

import HOPS
import SebbuScience
import PythonKitUtilities

private func masterEquationSolution(endTime: Double, initialRho: Matrix<Complex<Double>>, H: Matrix<Complex<Double>>, gamma: Double, O: Matrix<Complex<Double>>) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let ODagger = O.conjugateTranspose
    let ODaggerO = ODagger.dot(O)
    var solver = RK45FixedStep(initialState: initialRho, t0: 0.0, dt: 0.01) { t, rho in
        var result = -.i * (H.dot(rho) - rho.dot(H))
        result += gamma * O.dot(rho.dot(ODagger))
        result -= 0.5 * gamma * (ODaggerO.dot(rho) + rho.dot(ODaggerO))
        return result
    }
    var tSpace: [Double] = []
    var rho: [Matrix<Complex<Double>>] = []
    while solver.t < endTime {
        let (t, _rho) = solver.step()
        tSpace.append(t)
        rho.append(_rho)
    }
    return (tSpace, rho)
}

public func radiativeDampingExample(realizations: Int, endTime: Double = 7.0) {
    let A = 0.000027
    let omegaC = 1.447
    let gamma = 0.0175
    
    let (G, W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        //let (G, W) = tPFD.fit(x: tSpace, y: bcf, realTerms: 4, imaginaryTerms: 4)
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let hierarchy = HOPSHierarchy(dimension: 2, L: L, G: G, W: W, depth: 3)
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, Complex(0.175 / 2), Complex(0.175 / 2), Complex(0.1)], rows: 2, columns: 2)
//    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
//        spectralDensity(omega: omega, A: A, omegaC: omegaC)
//    }
    let zGenerator = ZeroNoiseProcessGenerator()
    let whiteNoiseGenerator = PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gamma / 2, start: 0, end: endTime, step: 0.01)
    
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    let sigmaMinus: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaPlus = sigmaMinus.conjugateTranspose
    let sigmaPlusSigmaMinus = sigmaPlus.dot(sigmaMinus)
    
    let gammaSigmaPlusSigmaMinus = -0.5 * gamma * sigmaPlusSigmaMinus
    
    let batchSize = 64
    
    let linearStart = ContinuousClock().now
    var trajectoriesComputed = 0
    var linearTSpace: [Double] = []
    var linearRho: [Matrix<Complex<Double>>] = []
    while trajectoriesComputed < realizations {
        let trajectoriesToCompute = Swift.min(realizations - trajectoriesComputed, batchSize)
        trajectoriesComputed += trajectoriesToCompute
        let batchComputationTime = ContinuousClock().measure {
            let noises = zGenerator.generateParallel(count: trajectoriesToCompute)
            let whiteNoises = whiteNoiseGenerator.generateParallel(count: trajectoriesToCompute)
            let linearTrajectories = zip(noises, whiteNoises).parallelMap { z, w in
                let customOperator: @Sendable (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
                    gammaSigmaPlusSigmaMinus
                }
                return hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z, whiteNoise: w, diffusionOperator: sigmaMinus, customOperators: [customOperator], stepSize: 0.01)
            }
            
            for (tSpace, trajectory) in linearTrajectories {
                let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory)
                if linearRho.isEmpty {
                    linearTSpace = tSpace
                    linearRho = _rho.map { $0 / Double(realizations) }
                } else {
                    for i in 0..<_rho.count {
                        linearRho[i].add(_rho[i], multiplied: 1.0 / Double(realizations))
                    }
                }
            }
        }
        print("[Linear]: \(batchComputationTime * Double(realizations - trajectoriesComputed) / Double(batchSize)) left")
    }
    let linearEnd = ContinuousClock().now
    
    let nonLinearStart = ContinuousClock().now
    trajectoriesComputed = 0
    var nonLinearTSpace: [Double] = []
    var nonLinearRho: [Matrix<Complex<Double>>] = []
    while trajectoriesComputed < realizations {
        let trajectoriesToCompute = Swift.min(realizations - trajectoriesComputed, batchSize)
        trajectoriesComputed += trajectoriesToCompute
        
        let batchComputationTime = ContinuousClock().measure {
            let noises = zGenerator.generateParallel(count: trajectoriesToCompute)
            let whiteNoises = whiteNoiseGenerator.generateParallel(count: trajectoriesToCompute)
            let nonLinearTrajectories = zip(noises, whiteNoises).parallelMap { z, w in
                nonisolated(unsafe) var _O = sigmaMinus
                let customOperator: (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
                    _O.copyElements(from: gammaSigmaPlusSigmaMinus)
                    let factor = gamma * state.inner(state, metric: sigmaPlus) / state.normSquared
                    _O.add(sigmaMinus, multiplied: factor)
                    return _O
                }
                return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, whiteNoise: w, diffusionOperator: sigmaMinus, customOperators: [customOperator], stepSize: 0.01)
            }
            for (tSpace, trajectory) in nonLinearTrajectories {
                let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory, normalize: true)
                if nonLinearRho.isEmpty {
                    nonLinearTSpace = tSpace
                    nonLinearRho = _rho.map { $0 / Double(realizations) }
                } else {
                    for i in 0..<_rho.count {
                        nonLinearRho[i].add(_rho[i], multiplied: 1.0 / Double(realizations))
                    }
                }
            }
        }
        print("[Non-linear]: \(batchComputationTime * Double(realizations - trajectoriesComputed) / Double(batchSize))")
    }
    let nonLinearEnd = ContinuousClock().now
    print("Linear time: \(linearEnd - linearStart)")
    print("Non-linear time: \(nonLinearEnd - nonLinearStart)")
    
    let (masterEquationTSpace, rho) = masterEquationSolution(endTime: endTime, initialRho: initialState.outer(initialState), H: H, gamma: gamma, O: sigmaMinus)
    
    let linearX = linearRho.map { 2 * $0[0, 1].real }
    let linearY = linearRho.map { 2 * $0[0, 1].imaginary }
    let linearZ = linearRho.map { $0[0, 0].real - $0[1, 1].real }
    
    let nonLinearX = nonLinearRho.map { 2 * $0[0, 1].real }
    let nonLinearY = nonLinearRho.map { 2 * $0[0, 1].imaginary }
    let nonLinearZ = nonLinearRho.map { $0[0, 0].real - $0[1, 1].real }
    
    let masterEquationX = rho.map { 2 * $0[0, 1].real }
    let masterEquationY = rho.map { 2 * $0[0, 1].imaginary }
    let masterEquationZ = rho.map { $0[0, 0].real - $0[1, 1].real }
    
    plt.figure()
//    plt.plot(x: linearTSpace, y: linearX, label: "Lin <x>")
//    plt.plot(x: linearTSpace, y: linearY, label: "Lin <y>")
//    plt.plot(x: linearTSpace, y: linearZ, label: "Lin <z>")
    
    plt.plot(x: nonLinearTSpace, y: nonLinearX, label: "Non-lin <x>", linestyle: "--")
    plt.plot(x: nonLinearTSpace, y: nonLinearY, label: "Non-lin <y>", linestyle: "--")
    plt.plot(x: nonLinearTSpace, y: nonLinearZ, label: "Non-lin <z>", linestyle: "--")
    
    plt.plot(x: masterEquationTSpace, y: masterEquationX, label: "Master Eq. <x>", linestyle: "-.")
    plt.plot(x: masterEquationTSpace, y: masterEquationY, label: "Master Eq. <y>", linestyle: "-.")
    plt.plot(x: masterEquationTSpace, y: masterEquationZ, label: "Master Eq. <z>", linestyle: "-.")
    
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("rho")
    plt.show()
    plt.close()
}

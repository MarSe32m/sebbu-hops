//
//  RadiativeDamping.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 15.10.2025.
//

import HOPS
import SebbuScience
import PythonKitUtilities

private func masterEquationSolution(endTime: Double, initialRho: Matrix<Complex<Double>>, H: Matrix<Complex<Double>>, channels: [(gamma: Double, L: Matrix<Complex<Double>>)]) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let channels = channels.map { gamma, L in (gamma, L, L.conjugateTranspose, L.conjugateTranspose.dot(L))}
    var solver = RK45FixedStep(initialState: initialRho, t0: 0.0, dt: 0.01) { t, rho in
        var result = -.i * (H.dot(rho) - rho.dot(H))
        for (rate, L, LDagger, LDaggerL) in channels {
            result += rate * L.dot(rho.dot(LDagger))
            result -= 0.5 * rate * (LDaggerL.dot(rho) + rho.dot(LDaggerL))
        }
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

public func radiativeDampingPlusPumpingExample(realizations: Int, endTime: Double = 7.0) {
    let A = 0.000027
    let omegaC = 1.447
    let gammaR = 0.0175
    let gammaP = 0.0175 / 2
    
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
    
    let H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(0.1)], rows: 2, columns: 2)
    let zGenerator = GaussianFFTNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC)
    }
    //let zGenerator = ZeroNoiseProcessGenerator()
    let whiteNoiseRGenerator = PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaR / 2, start: 0, end: endTime, step: 0.01)
    let whiteNoisePGenerator = PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaP / 2, start: 0, end: endTime, step: 0.01)
    
    let initialState: Vector<Complex<Double>> = [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())]
    
    let sigmaMinus: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaPlus = sigmaMinus.conjugateTranspose
    let sigmaPlusSigmaMinus = sigmaPlus.dot(sigmaMinus)
    let sigmaMinusSigmaPlus = sigmaMinus.dot(sigmaPlus)
    
    let gammaRSigmaPlusSigmaMinus = -0.5 * gammaR * sigmaPlusSigmaMinus
    let gammaPSigmaMinusSigmaPlus = -0.5 * gammaP * sigmaMinusSigmaPlus
    
    let _operatorSum = gammaRSigmaPlusSigmaMinus + gammaPSigmaMinusSigmaPlus
    
    let batchSize = 100
    
    let linearStart = ContinuousClock().now
    var trajectoriesComputed = 0
    var linearTSpace: [Double] = []
    var linearRho: [Matrix<Complex<Double>>] = []
    while trajectoriesComputed < realizations {
        let trajectoriesToCompute = Swift.min(realizations - trajectoriesComputed, batchSize)
        trajectoriesComputed += trajectoriesToCompute
        let batchComputationTime = ContinuousClock().measure {
            let noises = zGenerator.generateParallel(count: trajectoriesToCompute / 2)
            let whiteNoisesR = whiteNoiseRGenerator.generateParallel(count: trajectoriesToCompute / 2)
            let whiteNoisesP = whiteNoisePGenerator.generateParallel(count: trajectoriesToCompute / 2)
            let linearTrajectories = zip(zip(noises, whiteNoisesR), whiteNoisesP).parallelMap { zwR, wP in
                let z = zwR.0
                let wR = zwR.1
                let customOperator: @Sendable (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
                    _operatorSum
                }
                
                return hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z,
                                             whiteNoises: [wR, wP],
                                             diffusionOperators: [sigmaMinus, sigmaPlus], customOperators: [customOperator], stepSize: 0.1)
            }
            let linearAntitheticTrajectories = zip(zip(noises, whiteNoisesR), whiteNoisesP).parallelMap { zwR, wP in
                let z = zwR.0
                //let z = zwR.0.antithetic()
                let wR = zwR.1.antithetic()
                let wP = wP.antithetic()
                let customOperator: @Sendable (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
                    _operatorSum
                }
                
                return hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z,
                                             whiteNoises: [wR, wP],
                                             diffusionOperators: [sigmaMinus, sigmaPlus], customOperators: [customOperator], stepSize: 0.1)
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
            for (tSpace, trajectory) in linearAntitheticTrajectories {
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
            let noises = zGenerator.generateParallel(count: trajectoriesToCompute / 2)
            let whiteNoisesR = whiteNoiseRGenerator.generateParallel(count: trajectoriesToCompute / 2)
            let whiteNoisesP = whiteNoisePGenerator.generateParallel(count: trajectoriesToCompute / 2)
            let nonLinearTrajectories = zip(zip(noises, whiteNoisesR), whiteNoisesP).parallelMap { zwR, wP in
                let z = zwR.0
                let wR = zwR.1
                nonisolated(unsafe) var _O = sigmaMinus
                let customOperator: (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
                    _O.copyElements(from: _operatorSum)
                    var factor = gammaR * state.inner(state, metric: sigmaPlus) / state.normSquared
                    _O.add(sigmaMinus, multiplied: factor)
                    factor = gammaP * state.inner(state, metric: sigmaMinus) / state.normSquared
                    _O.add(sigmaPlus, multiplied: factor)
                    return _O
                }
                return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z,
                                                whiteNoises: [wR, wP],
                                                diffusionOperators: [sigmaMinus, sigmaPlus], customOperators: [customOperator], stepSize: 0.1)
            }
            let nonLinearAntitheticTrajectories = zip(zip(noises, whiteNoisesR), whiteNoisesP).parallelMap { zwR, wP in
                //let z = zwR.0.antithetic()
                let z = zwR.0
                let wR = zwR.1.antithetic()
                let wP = wP.antithetic()
                nonisolated(unsafe) var _O = sigmaMinus
                let customOperator: (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
                    _O.copyElements(from: _operatorSum)
                    var factor = gammaR * state.inner(state, metric: sigmaPlus) / state.normSquared
                    _O.add(sigmaMinus, multiplied: factor)
                    factor = gammaP * state.inner(state, metric: sigmaMinus) / state.normSquared
                    _O.add(sigmaPlus, multiplied: factor)
                    return _O
                }
                return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z,
                                                whiteNoises: [wR, wP],
                                                diffusionOperators: [sigmaMinus, sigmaPlus], customOperators: [customOperator], stepSize: 0.1)
            }
            for (tSpace, trajectory) in nonLinearTrajectories {
                let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory, normalized: true)
                if nonLinearRho.isEmpty {
                    nonLinearTSpace = tSpace
                    nonLinearRho = _rho.map { $0 / Double(realizations) }
                } else {
                    for i in 0..<_rho.count {
                        nonLinearRho[i].add(_rho[i], multiplied: 1.0 / Double(realizations))
                    }
                }
            }
            for (tSpace, trajectory) in nonLinearAntitheticTrajectories {
                let _rho = hierarchy.mapTrajectoryToDensityMatrix(trajectory, normalized: true)
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
    let channels = [(gammaR, sigmaMinus), (gammaP, sigmaPlus)]
    let (masterEquationTSpace, rho) = masterEquationSolution(endTime: endTime, initialRho: initialState.outer(initialState), H: H, channels: channels)
    
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

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

public func radiativeDampingPlusPumpingMultiParticleExample(realizations: Int, endTime: Double = 7.0) {
    let A = 0.000027
    let omegaC = 1.447
    let gammaR = 0.0175
    let gammaP = 0.0175 / 2
    
    let (_G, _W) = {
        let T: Double = .zero
        let tSpace = [Double].linearSpace(0, 10, 501)
        let bcf = tSpace.map { bathCorrelationFunction(A: A, omegaC: omegaC, t: $0, temperature: T) }
        //let (G, W) = tPFD.fit(x: tSpace, y: bcf, realTerms: 4, imaginaryTerms: 4)
        let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: 3)
        return (G, W)
    }()
    
    let G = [[_G, []],
             [[], _G]]
    let W = [[_W, []],
             [[], _W]]
    
    let L: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, .one], rows: 2, columns: 2)
    let L1 = L.kronecker(.identity(rows: 2))
    let L2 = Matrix<Complex<Double>>.identity(rows: 2).kronecker(L)
    let hierarchy = HOPSMultiParticleHierarchy(dimension: 4, L: [L1, L2], G: G, W: W, depth: 3)
    
    let _H: Matrix<Complex<Double>> = .init(elements: [.zero, .zero, .zero, Complex(0.1)], rows: 2, columns: 2)
    let H = _H.kronecker(.identity(rows: 2)) + .identity(rows: 2).kronecker(_H)
    
    let zGenerator = GaussianFFTMultiNoiseProcessGenerator(tMax: endTime) { omega in
        spectralDensity(omega: omega, A: A, omegaC: omegaC) * .identity(rows: 2)
    }
    
    //let zGenerator = ZeroNoiseProcessGenerator()
    let radiativeDampingWhiteNoiseGenerators = (0..<2).map { _ in PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaR / 2, start: 0, end: endTime, step: 0.01) }
    let radiativePumpingWhiteNoiseGenerators = (0..<2).map { _ in PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaP / 2, start: 0, end: endTime, step: 0.01) }
    
    let _initialState: Matrix<Complex<Double>> = .init(elements: [Complex((0.5).squareRoot()), Complex((0.5).squareRoot())], rows: 2, columns: 1)
    let initialState: Vector<Complex<Double>> = Vector(_initialState.kronecker(_initialState).elements)
    
    let sigmaMinus: Matrix<Complex<Double>> = .init(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaPlus = sigmaMinus.conjugateTranspose
    
    let sigmaPlusSigmaMinus = sigmaPlus.dot(sigmaMinus)
    let sigmaMinusSigmaPlus = sigmaMinus.dot(sigmaPlus)
    
    let sigmaPlus1: Matrix<Complex<Double>> = sigmaPlus.kronecker(.identity(rows: 2))
    let sigmaPlus2: Matrix<Complex<Double>> = .identity(rows: 2).kronecker(sigmaPlus)
    let sigmaMinus1 = sigmaPlus1.conjugateTranspose
    let sigmaMinus2 = sigmaPlus2.conjugateTranspose
    
    let gammaRSigmaPlusSigmaMinus = -0.5 * gammaR * sigmaPlusSigmaMinus
    let gammaPSigmaMinusSigmaPlus = -0.5 * gammaP * sigmaMinusSigmaPlus
    
    let _operatorSum = (gammaRSigmaPlusSigmaMinus + gammaPSigmaMinusSigmaPlus).kronecker(.identity(rows: 2)) + .identity(rows: 2).kronecker(gammaRSigmaPlusSigmaMinus + gammaPSigmaMinusSigmaPlus)
    
    let diffusionOperators = [sigmaMinus.kronecker(.identity(rows: 2)), .identity(rows: 2).kronecker(sigmaMinus),
                              sigmaPlus.kronecker(.identity(rows: 2)), .identity(rows: 2).kronecker(sigmaPlus)]
    
    let batchSize = 256
    
    let linearStart = ContinuousClock().now
    var trajectoriesComputed = Swift.max(16, realizations - 16)
    var linearTSpace: [Double] = []
    var linearRho: [Matrix<Complex<Double>>] = []
    while trajectoriesComputed < realizations {
        let trajectoriesToCompute = Swift.min(realizations - trajectoriesComputed, batchSize)
        trajectoriesComputed += trajectoriesToCompute
        let batchComputationTime = ContinuousClock().measure {
            let trajectories = (0..<trajectoriesToCompute / 2).parallelMap { _ in
                let z = zGenerator.generate()
                let wR = radiativeDampingWhiteNoiseGenerators.map { $0.generate() }
                let wP = radiativePumpingWhiteNoiseGenerators.map { $0.generate() }
                let customOperator: @Sendable (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { _, _ in
                    _operatorSum
                }
                let trajectory = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z,
                                                         whiteNoises: wR + wP,
                                                         diffusionOperators: diffusionOperators, customOperators: [customOperator], stepSize: 0.1)
                let antitheticTrajectory = hierarchy.solveLinear(end: endTime, initialState: initialState, H: H, z: z.map { $0.antithetic() },
                                                                   whiteNoises: (wR + wP).map { $0.antithetic() },
                                                                   diffusionOperators: diffusionOperators, customOperators: [customOperator], stepSize: 0.1)
                return (trajectory, antitheticTrajectory)
                
            }
            for (trajectory, antitheticTrajectory) in trajectories {
                let _rho1 = hierarchy.mapTrajectoryToDensityMatrix(trajectory.trajectory)
                let _rho2 = hierarchy.mapTrajectoryToDensityMatrix(antitheticTrajectory.trajectory)
                let _rho = zip(_rho1, _rho2).map { 0.5 * ($0 + $1) }
                if linearRho.isEmpty {
                    linearTSpace = trajectory.tSpace
                    linearRho = _rho.map { 2.0 * $0 / Double(realizations) }
                } else {
                    for i in 0..<_rho.count {
                        linearRho[i].add(_rho[i], multiplied: 2.0 / Double(realizations))
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
            let trajectories = (0..<trajectoriesToCompute / 2).parallelMap { _ in
                let z = zGenerator.generate()
                let wR = radiativeDampingWhiteNoiseGenerators.map { $0.generate() }
                let wP = radiativePumpingWhiteNoiseGenerators.map { $0.generate() }
                var _O = _operatorSum
                let customOperator: (Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>> = { t, state in
                    _O.copyElements(from: _operatorSum)
                    let normSquared = state.normSquared
                    // Radiative damping
                    var factor = gammaR * state.inner(state, metric: sigmaPlus1) / normSquared
                    _O.add(sigmaMinus1, multiplied: factor)
                    factor = gammaR * state.inner(state, metric: sigmaPlus2) / normSquared
                    _O.add(sigmaMinus2, multiplied: factor)
                    
                    // Pumping
                    factor = gammaP * state.inner(state, metric: sigmaMinus1) / normSquared
                    _O.add(sigmaPlus1, multiplied: factor)
                    factor = gammaP * state.inner(state, metric: sigmaMinus2) / normSquared
                    _O.add(sigmaPlus2, multiplied: factor)
                    return _O
                }
                let trajectory = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z,
                                                         whiteNoises: wR + wP,
                                                         diffusionOperators: diffusionOperators, customOperators: [customOperator], stepSize: 0.1)
                let antitheticTrajectory = hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z.map { $0.antithetic() },
                                                                   whiteNoises: (wR + wP).map { $0.antithetic() },
                                                                   diffusionOperators: diffusionOperators, customOperators: [customOperator], stepSize: 0.1)
                return (trajectory, antitheticTrajectory)
                
            }
            for (trajectory, antitheticTrajectory) in trajectories {
                let _rho1 = hierarchy.mapTrajectoryToDensityMatrix(trajectory.trajectory, normalize: true)
                let _rho2 = hierarchy.mapTrajectoryToDensityMatrix(antitheticTrajectory.trajectory, normalize: true)
                let _rho = zip(_rho1, _rho2).map { 0.5 * ($0 + $1) }
                if nonLinearRho.isEmpty {
                    nonLinearTSpace = trajectory.tSpace
                    nonLinearRho = _rho.map { 2.0 * $0 / Double(realizations) }
                } else {
                    for i in 0..<_rho.count {
                        nonLinearRho[i].add(_rho[i], multiplied: 2.0 / Double(realizations))
                    }
                }
            }
        }
        print("[Non-linear]: \(batchComputationTime * Double(realizations - trajectoriesComputed) / Double(batchSize))")
    }
    let nonLinearEnd = ContinuousClock().now
    print("Linear time: \(linearEnd - linearStart)")
    print("Non-linear time: \(nonLinearEnd - nonLinearStart)")
    let channels = [(gammaR, sigmaMinus1), (gammaR, sigmaMinus2),
                    (gammaP, sigmaPlus1), (gammaP, sigmaPlus2)]
    let (masterEquationTSpace, rho) = masterEquationSolution(endTime: endTime, initialRho: initialState.outer(initialState), H: H, channels: channels)
    
    let linearRho1 = linearRho.map { MatrixOperations.partialTrace($0, dimensions: [2,2], keep: [0]) }
    let nonLinearRho1 = nonLinearRho.map { MatrixOperations.partialTrace($0, dimensions: [2,2], keep: [0]) }
    let rho1 = rho.map { MatrixOperations.partialTrace($0, dimensions: [2,2], keep: [0]) }
    plotBloch(linearHOPS: (linearTSpace, linearRho1), nonLinearHOPS: (nonLinearTSpace, nonLinearRho1), masterEquation: (masterEquationTSpace, rho1), title: "System 1")
    
    let linearRho2 = linearRho.map { MatrixOperations.partialTrace($0, dimensions: [2,2], keep: [1]) }
    let nonLinearRho2 = nonLinearRho.map { MatrixOperations.partialTrace($0, dimensions: [2,2], keep: [1]) }
    let rho2 = rho.map { MatrixOperations.partialTrace($0, dimensions: [2,2], keep: [1]) }
    plotBloch(linearHOPS: (linearTSpace, linearRho2), nonLinearHOPS: (nonLinearTSpace, nonLinearRho2), masterEquation: (masterEquationTSpace, rho2), title: "System 2")
}


private func plotBloch(linearHOPS: (tSpace: [Double], rho: [Matrix<Complex<Double>>]),
                       nonLinearHOPS: (tSpace: [Double], rho: [Matrix<Complex<Double>>]),
                       masterEquation: (tSpace: [Double], rho: [Matrix<Complex<Double>>]),
                       title: String) {
//    let linearX = linearHOPS.rho.map { 2 * $0[0, 1].real }
//    let linearY = linearHOPS.rho.map { 2 * $0[0, 1].imaginary }
//    let linearZ = linearHOPS.rho.map { $0[0, 0].real - $0[1, 1].real }
//    
    let nonLinearX = nonLinearHOPS.rho.map { 2 * $0[0, 1].real }
    let nonLinearY = nonLinearHOPS.rho.map { 2 * $0[0, 1].imaginary }
    let nonLinearZ = nonLinearHOPS.rho.map { $0[0, 0].real - $0[1, 1].real }
    
    let masterEquationX = masterEquation.rho.map { 2 * $0[0, 1].real }
    let masterEquationY = masterEquation.rho.map { 2 * $0[0, 1].imaginary }
    let masterEquationZ = masterEquation.rho.map { $0[0, 0].real - $0[1, 1].real }
    
    plt.figure()
//    plt.plot(x: linearHOPS.tSpace, y: linearX, label: "Lin <x>")
//    plt.plot(x: linearHOPS.tSpace, y: linearY, label: "Lin <y>")
//    plt.plot(x: linearHOPS.tSpace, y: linearZ, label: "Lin <z>")
    
    plt.plot(x: nonLinearHOPS.tSpace, y: nonLinearX, label: "Non-lin <x>", linestyle: "--")
    plt.plot(x: nonLinearHOPS.tSpace, y: nonLinearY, label: "Non-lin <y>", linestyle: "--")
    plt.plot(x: nonLinearHOPS.tSpace, y: nonLinearZ, label: "Non-lin <z>", linestyle: "--")
    
    plt.plot(x: masterEquation.tSpace, y: masterEquationX, label: "Master Eq. <x>", linestyle: "-.")
    plt.plot(x: masterEquation.tSpace, y: masterEquationY, label: "Master Eq. <y>", linestyle: "-.")
    plt.plot(x: masterEquation.tSpace, y: masterEquationZ, label: "Master Eq. <z>", linestyle: "-.")
    
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("rho")
    plt.title(title)
    plt.show()
    plt.close()
}

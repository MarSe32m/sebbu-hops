//
//  DrivenDissipativeCavityMode.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 8.5.2026.
//

import SebbuScience
import DequeModule
import HOPS
import PythonKit
import PythonKitUtilities

private func commutator(_ A: Matrix<Complex<Double>>, _ B: Matrix<Complex<Double>>, multiplied: Complex<Double>, into: inout Matrix<Complex<Double>>) {
    A._dot(B, multiplied: multiplied, into: &into)
    B._dot(A, multiplied: -multiplied, addingInto: &into)
}

private func antiCommutator(_ A: Matrix<Complex<Double>>, _ B: Matrix<Complex<Double>>, multiplied: Complex<Double>, addingInto: inout Matrix<Complex<Double>>) {
    A.dot(B, multiplied: multiplied, addingInto: &addingInto)
    B.dot(A, multiplied: multiplied, addingInto: &addingInto)
}

private func solveMasterEquation(start: Double, end: Double, initialState: Matrix<Complex<Double>>, H: Matrix<Complex<Double>>, L: [Matrix<Complex<Double>>], rates: [Double], stepSize: Double) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let LDagger = L.map { $0.conjugateTranspose }
    let LDaggerL = L.map { $0.conjugateTranspose.dot($0) }
    var scratch: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
    var rhoCache: Deque<Matrix<Complex<Double>>> = .init(repeating: scratch, count: 4)
    var solver = RK4Solver(initialState: initialState, t0: start, dt: stepSize) { t, rho in
        var result = rhoCache.removeFirst()
        defer { rhoCache.append(result) }
        commutator(H, rho, multiplied: -.i, into: &result)
        for i in rates.indices {
            let L = L[i]
            let LDagger = LDagger[i]
            let LDaggerL = LDaggerL[i]
            let rate = rates[i]
            L.dot(rho, into: &scratch)
            scratch.dot(LDagger, multiplied: Complex(rate), addingInto: &result)
            antiCommutator(LDaggerL, rho, multiplied: -Complex(0.5 * rate), addingInto: &result)
        }
        return result
    }
    var tSpace: [Double] = []
    var densityMatrix: [Matrix<Complex<Double>>] = []
    while solver.t < end {
        let (t, rho) = solver.step()
        tSpace.append(t)
        densityMatrix.append(rho)
    }
    return (tSpace, densityMatrix)
}


private func dagger(_ O: Matrix<Complex<Double>>) -> Matrix<Complex<Double>> {
    O.conjugateTranspose
}

private func projector(dimension: Int, _ i: Int, _ j: Int) -> Matrix<Complex<Double>> {
    var O: Matrix<Complex<Double>> = .zeros(rows: dimension, columns: dimension)
    O[i, j] = .one
    return O
}

private func tensor(_ A: Matrix<Complex<Double>>...) -> Matrix<Complex<Double>> {
    if A.isEmpty { return .zeros(rows: 0, columns: 0) }
    if A.count == 1 { return A.first! }
    var result = A[0]
    for i in 1..<A.count {
        result = result.kronecker(A[i])
    }
    return result
}

private func tensor(_ A: [Matrix<Complex<Double>>]) -> Matrix<Complex<Double>> {
    if A.isEmpty { return .zeros(rows: 0, columns: 0) }
    if A.count == 1 { return A.first! }
    var result = A[0]
    for i in 1..<A.count {
        result = result.kronecker(A[i])
    }
    return result
}

private func tensor(_ A: [Matrix<Complex<Double>>], _ B: Matrix<Complex<Double>>) -> Matrix<Complex<Double>> {
    if A.isEmpty { return B }
    var result = A[0]
    for i in 1..<A.count {
        result = result.kronecker(A[i])
    }
    result = result.kronecker(B)
    return result
}

private func tensor(_ A: Matrix<Complex<Double>>, _ B: [Matrix<Complex<Double>>], _ C: Matrix<Complex<Double>>...) -> Matrix<Complex<Double>> {
    var result = A
    for b in B { result = result.kronecker(b) }
    for c in C { result = result.kronecker(c) }
    return result
}

private func systemOperators(dimension: Int) -> Matrix<Matrix<Complex<Double>>> {
    var result: [Matrix<Complex<Double>>] = []
    for i in 0..<dimension {
        for j in 0..<dimension {
            result.append(projector(dimension: dimension, i, j))
        }
    }
    return .init(elements: result, rows: dimension, columns: dimension)
}


private func cavityOperators(dimension: Int) -> (a: Matrix<Complex<Double>>, aDagger: Matrix<Complex<Double>>, num: Matrix<Complex<Double>>) {
    var a: Matrix<Complex<Double>> = .zeros(rows: dimension, columns: dimension)
    for i in 1..<dimension {
        a[i - 1, i] = Complex(Double(i).squareRoot())
    }
    let aDagger = a.conjugateTranspose
    let num = aDagger.dot(a)
    return (a, aDagger, num)
}

private func solveModelWithMasterEquation(endTime: Double, omegaX: Double, g: Double, omegaC: Double, gammaMinus: Double, gammaPlus: Double) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let cavityDimension = 15
    let (a, adag, num) = cavityOperators(dimension: cavityDimension)
    let IC = Matrix<Complex<Double>>.identity(rows: a.rows)
    let systemOperators = systemOperators(dimension: 2)
    let IM = Matrix<Complex<Double>>.identity(rows: 2)
    
    let H = /* 0.5 * tensor(systemOperators[0, 1] + systemOperators[1, 0], IC) + */ omegaX * tensor(systemOperators[1, 1], IC)
            + omegaC * tensor(IM, num)
            + g * (tensor(systemOperators[0, 1], adag) + tensor(systemOperators[1, 0], a))
    let systemInitialVector: Vector<Complex<Double>> = [Complex(.sqrt(0.5)), Complex(.sqrt(0.5))]
    let systemInitial = systemInitialVector.outer(systemInitialVector.conjugate)
    var cavityInitial: Matrix<Complex<Double>> = .zeros(rows: cavityDimension, columns: cavityDimension)
    let neff = gammaPlus / (gammaMinus - gammaPlus)
    //let q = gammaPlus / gammaMinus
    //var qn = 1.0
    for i in 0..<cavityDimension {
        cavityInitial[i, i] = Complex(.pow(neff, i) / .pow(neff + 1, i + 1))
        //qn *= q
    }
    cavityInitial /= cavityInitial.trace.real
    //let cavityInitial: Matrix<Complex<Double>> = .init(elements: [1] + .init(repeating: 0, count: cavityDimension - 1), rows: cavityDimension, columns: 1)
    let initialRho = tensor(systemInitial, cavityInitial)
    //let initialStateVector = Vector(systemInitial.kronecker(cavityInitial).elements)
    //let initialRho = initialStateVector.outer(initialStateVector.conjugate)
    
    let (tSpace, rho) = solveMasterEquation(start: 0, end: endTime, initialState: initialRho, H: H, L: [tensor(IM, a), tensor(IM, adag)], rates: [gammaMinus, gammaPlus], stepSize: 0.01)
    return (tSpace, rho.map { MatrixOperations.partialTrace($0, dimensions: [2, cavityDimension], keep: [0])})
}

private func _growCapacity(_ capacity: Int) -> Int {
    // 13/8 = 1.625x growth for storage
    return capacity &+ (capacity &>> 1) &+ (capacity &>> 3)
}

private func solveModelWithHOPS(endTime: Double, omegaX: Double, g: Double, omegaC: Double, gammaMinus: Double, gammaPlus: Double, depth: Int, trajectories: Int) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let gamma = gammaMinus - gammaPlus
    let neff = gammaPlus / gamma
    let G1 = g * g * (1 + neff)
    let G2 = g * g * neff
    let W1 = Complex(gamma / 2, omegaC)
    let W2 = Complex(gamma / 2, -omegaC)
    
//    let generator2 = GaussianFFTNoiseProcessGenerator(tMax: endTime, dtMax: 0.01) { omega in
//        G / .pi * (1.0 / (W - Complex(imaginary: omega))).real
//    }
    
    let vaccuumNoiseGenerator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G1, W: W1, start: 0, end: endTime, step: 0.005)
    let temperatureGenerator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G2, W: W2, start: 0, end: endTime, step: 0.005)
//    let temperatureGenerator = ZeroNoiseProcessGenerator()
    
    let sigmaMinus = Matrix<Complex<Double>>(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaPlus = Matrix<Complex<Double>>(elements: [.zero, .zero, .one, .zero], rows: 2, columns: 2)
    let H = Matrix<Complex<Double>>(elements: [.zero, Complex(0), Complex(0), Complex(omegaX)], rows: 2, columns: 2)
    let initialState: Vector<Complex<Double>> = [Complex(.sqrt(0.5)), Complex(.sqrt(0.5))]
    
    //let hierarchy = HOPSHierarchy(dimension: 2, L: sigmaMinus, G: [Complex(G)], W: [W], depth: depth)
    let hierarchy = HOPSMultiParticleHierarchy(dimension: 2,
                                               L: [sigmaMinus, sigmaPlus],
                                               G: [[[Complex(G1)], []],
                                                [[], [Complex(G2)]]],
                                               W: [[[W1], []],
                                                  [[], [W2]]], depth: depth)
    
    
    let weight = Double(trajectories).reciprocal!
    var trajectoriesComputed = 0
    var rho: [Matrix<Complex<Double>>] = []
    var tSpace: [Double] = []
    while trajectoriesComputed < trajectories {
        let batch = Swift.min(100, trajectories - trajectoriesComputed)
        trajectoriesComputed += batch
        print(trajectoriesComputed)
        let _trajectories = (0..<batch).parallelMap { _ in
            let z1 = vaccuumNoiseGenerator.generate()
            let z2 = temperatureGenerator.generate()
//            let z = generator2.generate()
//            let xi = temperatureGenerator.generate()
//            var O: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
//            let customOperator: ((Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>>) = { t, _ in
//                O.zeroElements()
//                let xiSample = xi(t)
//                O.add(sigmaMinus, multiplied: xiSample.conjugate)
//                O.add(sigmaPlus, multiplied: xiSample)
//                return O
//            }
            //return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: [z1, z2], shiftType: .meanField, stepSize: 0.01)
            return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: [z1, z2], stepSize: 0.01)
//            return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, shiftType: .meanField, customOperators: [customOperator], stepSize: 0.075)
        }
        for (_tSpace, _trajectory) in _trajectories {
            if tSpace.isEmpty {
                tSpace = _tSpace
            }
            let _rho = hierarchy.mapTrajectoryToDensityMatrix(_trajectory, normalize: true)
            if rho.isEmpty {
                rho = _rho.map { $0 / Double(trajectories) }
            } else {
                for i in rho.indices {
                    rho[i].add(_rho[i], multiplied: weight)
                }
            }
        }
    }
    return (tSpace, rho)
}

private func solveModelWithUnifiedHOPS(endTime: Double, omegaX: Double, g: Double, omegaC: Double, gammaMinus: Double, gammaPlus: Double, depth: Int, trajectories: Int) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let gamma = gammaMinus - gammaPlus
    let neff = gammaPlus / gamma
    let G1 = g * g * (1 + neff)
    let G2 = g * g * neff
    let W1 = Complex(gamma / 2, omegaC)
    let W2 = Complex(gamma / 2, -omegaC)
    
//    let generator2 = GaussianFFTNoiseProcessGenerator(tMax: endTime, dtMax: 0.01) { omega in
//        G / .pi * (1.0 / (W - Complex(imaginary: omega))).real
//    }
    
    let vaccuumNoiseGenerator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G1, W: W1, start: 0, end: endTime, step: 0.005)
    let temperatureGenerator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G2, W: W2, start: 0, end: endTime, step: 0.005)
//    let temperatureGenerator = ZeroNoiseProcessGenerator()
    
    let sigmaMinus = Matrix<Complex<Double>>(elements: [.zero, .one, .zero, .zero], rows: 2, columns: 2)
    let sigmaPlus = Matrix<Complex<Double>>(elements: [.zero, .zero, .one, .zero], rows: 2, columns: 2)
    let H = Matrix<Complex<Double>>(elements: [.zero, Complex(0), Complex(0), Complex(omegaX)], rows: 2, columns: 2)
    let initialState: Vector<Complex<Double>> = [Complex(.sqrt(0.5)), Complex(.sqrt(0.5))]
    
    let bcf1 = UnifiedHOPSHierarchy.BathCorrelationFunction(G: [Complex(G1)], W: [W1])
    let bcf2 = UnifiedHOPSHierarchy.BathCorrelationFunction(G: [Complex(G2)], W: [W2])
    let hierarchy = UnifiedHOPSHierarchy(dimension: 2, L: [sigmaMinus, sigmaPlus], bathCorrelationFunctions: [bcf1, bcf2], depth: depth)
//    let hierarchy = HOPSMultiParticleHierarchy(dimension: 2,
//                                               L: [sigmaMinus, sigmaPlus],
//                                               G: [[[Complex(G1)], []],
//                                                [[], [Complex(G2)]]],
//                                               W: [[[W1], []],
//                                                  [[], [W2]]], depth: depth)
    
    
    let weight = Double(trajectories).reciprocal!
    var trajectoriesComputed = 0
    var rho: [Matrix<Complex<Double>>] = []
    var tSpace: [Double] = []
    while trajectoriesComputed < trajectories {
        let batch = Swift.min(100, trajectories - trajectoriesComputed)
        trajectoriesComputed += batch
        print(trajectoriesComputed)
        let _trajectories = (0..<batch).parallelMap { _ in
            let z1 = vaccuumNoiseGenerator.generate()
            let z2 = temperatureGenerator.generate()
//            let z = generator2.generate()
//            let xi = temperatureGenerator.generate()
//            var O: Matrix<Complex<Double>> = .zeros(rows: 2, columns: 2)
//            let customOperator: ((Double, Vector<Complex<Double>>) -> Matrix<Complex<Double>>) = { t, _ in
//                O.zeroElements()
//                let xiSample = xi(t)
//                O.add(sigmaMinus, multiplied: xiSample.conjugate)
//                O.add(sigmaPlus, multiplied: xiSample)
//                return O
//            }
            //return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: [z1, z2], shiftType: .meanField, stepSize: 0.01)
            return hierarchy.solve(end: endTime, initialState: initialState, H: H, noises: [z1, z2].span, equationType: .nonLinear, shiftType: .meanField, stepSize: 0.01)
//            return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: [z1, z2], stepSize: 0.01)
//            return hierarchy.solveNonLinear(end: endTime, initialState: initialState, H: H, z: z, shiftType: .meanField, customOperators: [customOperator], stepSize: 0.075)
        }
        for trajectory in _trajectories {
            if tSpace.isEmpty { tSpace = trajectory.tSpace }
            let _rho = trajectory.densityMatrix(normalized: true)
            if rho.isEmpty {
                rho = _rho.map { $0 / Double(trajectories) }
            } else {
                for i in rho.indices {
                    rho[i].add(_rho[i], multiplied: weight)
                }
            }
        }
    }
    return (tSpace, rho)
}

private func solveModelWithQSD(endTime: Double, omegaX: Double, g: Double, omegaC: Double, gammaMinus: Double, gammaPlus: Double, trajectories: Int) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let cavityDimension = 10
    let (a, adag, num) = cavityOperators(dimension: cavityDimension)
    let IC = Matrix<Complex<Double>>.identity(rows: a.rows)
    let systemOperators = systemOperators(dimension: 2)
    let IM = Matrix<Complex<Double>>.identity(rows: 2)
    
    let H = omegaX * tensor(systemOperators[1, 1], IC) + omegaC * tensor(IM, num) + g * (tensor(systemOperators[0, 1], adag) + tensor(systemOperators[1, 0], a))
    let systemInitial: Matrix<Complex<Double>> = .init(elements: [Complex(.sqrt(0.5)), Complex(.sqrt(0.5))], rows: 2, columns: 1)
    let cavityInitial: Matrix<Complex<Double>> = .init(elements: [1] + .init(repeating: 0, count: cavityDimension - 1), rows: cavityDimension, columns: 1)
    let initialStateVector = Vector(systemInitial.kronecker(cavityInitial).elements)
    let qsdCalculation = QSDCalculation()
    
    let gammaMinusGenerator = PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaMinus / 2, start: 0, end: endTime, step: 0.01)
    
    let gammaPlusGenerator = PreSampledGaussianWhiteNoiseProcessGenerator(mean: 0, deviation: gammaPlus / 2, start: 0, end: endTime, step: 0.01)
    let LMinus = tensor(IM, a)
    let LPlus = tensor(IM, adag)
    
    let weight = Complex(Double(trajectories)).reciprocal!
    var trajectoriesComputed = 0
    var rho: [Matrix<Complex<Double>>] = []
    var tSpace: [Double] = []
    while trajectoriesComputed < trajectories {
        let batch = Swift.min(50, trajectories - trajectoriesComputed)
        trajectoriesComputed += batch
        print(trajectoriesComputed)
        let _trajectories = (0..<batch).parallelMap { _ in
            let gammaMinusOperator = QSDCalculation.JumpOperator(L: LMinus, rate: gammaMinus, noise: gammaMinusGenerator.generate())
            let gammaPlusOperator = QSDCalculation.JumpOperator(L: LPlus, rate: gammaPlus, noise: gammaPlusGenerator.generate())
            return qsdCalculation.solveNonLinear(end: endTime, initialState: initialStateVector, H: H, jumpOperators: [gammaPlusOperator, gammaMinusOperator], stepSize: 0.1)
        }
        for (_tSpace, _trajectory) in _trajectories {
            if tSpace.isEmpty {
                tSpace = _tSpace
            }
            let _rho = qsdCalculation.mapTrajectoryToDensityMatrix(_trajectory, normalize: true)
            if rho.isEmpty {
                rho = _rho.map { $0 / Double(trajectories) }
            } else {
                for i in rho.indices {
                    rho[i].add(_rho[i], multiplied: weight)
                }
            }
        }
    }
    return (tSpace, rho)
}

private func solveModelWithNMQSD(endTime: Double, omegaX: Double, g: Double, omegaC: Double, gammaMinus: Double, trajectories: Int) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let G = Complex(g * g)
    let W = Complex(gammaMinus / 2, omegaC)
    
    let gammaMinusGenerator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G.real, W: W, start: 0, end: endTime, step: 0.001)
    let weight = Double(trajectories).reciprocal!
    var trajectoriesComputed = 0
    var rho: [Matrix<Complex<Double>>] = []
    var tSpace: [Double] = []
    while trajectoriesComputed < trajectories {
        let batch = Swift.min(100, trajectories - trajectoriesComputed)
        trajectoriesComputed += batch
        print(trajectoriesComputed)
        let _trajectories = (0..<batch).parallelMap { _ in
            let z = gammaMinusGenerator.generate()
            let initialState: [Complex<Double>] = [Complex(.sqrt(0.5)), Complex(.sqrt(0.5)), .zero, .zero]
            var tSpace: [Double] = []
            var trajectory: [(Complex<Double>, Complex<Double>)] = []
            var resultCache: Deque<[Complex<Double>]> = Deque(repeating: [0,0,0,0], count: 4)
            var solver = RK4Solver(initialState: initialState, t0: 0, dt: 0.01) { t, state in
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                let sigmaPlusExpectation = state[0] * state[1].conjugate / (state[0].lengthSquared + state[1].lengthSquared)
                let shiftedNoise = z(t).conjugate + state[3]
                result[0] = shiftedNoise * state[1] + sigmaPlusExpectation * state[2] // d c_g(t) / dt
                result[1] = -.i * omegaX * state[1] - state[2] // d c_e(t) / dt
                result[2] = G * state[1] - W * state[2] // d m(t) / dt
                result[3] = G.conjugate * sigmaPlusExpectation - W.conjugate * state[3] // d xi(t) / dt
                return result
            }
            while solver.t < endTime {
                let (t, state) = solver.step()
                tSpace.append(t)
                trajectory.append((state[0], state[1]))
            }
            return (tSpace, trajectory)
        }
        for (_tSpace, _trajectory) in _trajectories {
            if tSpace.isEmpty {
                tSpace = _tSpace
            }
            let _rho = _trajectory.map { psi in
                let normSquared = psi.0.lengthSquared + psi.1.lengthSquared
                return Matrix<Complex<Double>>(elements: [Complex(psi.0.lengthSquared), psi.0 * psi.1.conjugate,
                                                          psi.0.conjugate * psi.1, Complex(psi.1.lengthSquared)], rows: 2, columns: 2) / normSquared
            }
            if rho.isEmpty {
                rho = _rho.map { $0 / Double(trajectories) }
            } else {
                for i in rho.indices {
                    rho[i].add(_rho[i], multiplied: weight)
                }
            }
        }
    }
    return (tSpace, rho)
}

private func solveModelWithOptimalHOPS(endTime: Double, omegaX: Double, g: Double, omegaC: Double, gammaMinus: Double, trajectories: Int) -> (tSpace: [Double], rho: [Matrix<Complex<Double>>]) {
    let G = Complex(g * g)
    let W = Complex(gammaMinus / 2, omegaC)
    
    let gammaMinusGenerator = PreSampledOrnsteinUhlenbeckProcessGenerator(G: G.real, W: W, start: 0, end: endTime, step: 0.001)
    let weight = Double(trajectories).reciprocal!
    var trajectoriesComputed = 0
    var rho: [Matrix<Complex<Double>>] = []
    var tSpace: [Double] = []
    while trajectoriesComputed < trajectories {
        let batch = Swift.min(100, trajectories - trajectoriesComputed)
        trajectoriesComputed += batch
        print(trajectoriesComputed)
        let _trajectories = (0..<batch).parallelMap { _ in
            let z = gammaMinusGenerator.generate()
            let initialState: [Complex<Double>] = [Complex(.sqrt(0.5)), Complex(.sqrt(0.5)), .zero, .zero]
            var tSpace: [Double] = []
            var trajectory: [(Complex<Double>, Complex<Double>)] = []
            var resultCache: Deque<[Complex<Double>]> = Deque(repeating: [0,0,0,0], count: 4)
            var solver = RK4Solver(initialState: initialState, t0: 0, dt: 0.01) { t, state in
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                let sigmaPlusExpectation = state[0] * state[1].conjugate / (state[0].lengthSquared + state[1].lengthSquared)
                let shiftedNoise = z(t).conjugate + state[3]
                result[0] = shiftedNoise * state[1] + sigmaPlusExpectation * state[2] // d c_g(t) / dt
                result[1] = -.i * omegaX * state[1] - state[2] // d c_e(t) / dt
                result[2] = G * state[1] - W * state[2] // d m(t) / dt
                result[3] = G.conjugate * sigmaPlusExpectation - W.conjugate * state[3] // d xi(t) / dt
                return result
            }
            while solver.t < endTime {
                let (t, state) = solver.step()
                tSpace.append(t)
                trajectory.append((state[0], state[1]))
            }
            return (tSpace, trajectory)
        }
        for (_tSpace, _trajectory) in _trajectories {
            if tSpace.isEmpty {
                tSpace = _tSpace
            }
            let _rho = _trajectory.map { psi in
                let normSquared = psi.0.lengthSquared + psi.1.lengthSquared
                return Matrix<Complex<Double>>(elements: [Complex(psi.0.lengthSquared), psi.0 * psi.1.conjugate,
                                                          psi.0.conjugate * psi.1, Complex(psi.1.lengthSquared)], rows: 2, columns: 2) / normSquared
            }
            if rho.isEmpty {
                rho = _rho.map { $0 / Double(trajectories) }
            } else {
                for i in rho.indices {
                    rho[i].add(_rho[i], multiplied: weight)
                }
            }
        }
    }
    return (tSpace, rho)
}

func drivenDissipativeCavityMode(endTime: Double, omegaX: Double, g: Double, omegaC: Double, gammaMinus: Double, gammaPlus: Double) {
    var start = ContinuousClock().now
    let (tSpaceME, rhoME) = solveModelWithMasterEquation(endTime: endTime, omegaX: omegaX, g: g, omegaC: omegaC, gammaMinus: gammaMinus, gammaPlus: gammaPlus)
    var end = ContinuousClock().now
    let xME = rhoME.map { 2 * $0[0, 1].real }
    let yME = rhoME.map { -2 * $0[0, 1].imaginary }
    let zME = rhoME.map { $0[0, 0] - $0[1, 1] }.real
    let METime = end - start
    
    start = .now
    let (tSpaceHOPS, rhoHOPS) = solveModelWithUnifiedHOPS(endTime: endTime, omegaX: omegaX, g: g, omegaC: omegaC, gammaMinus: gammaMinus, gammaPlus: gammaPlus, depth: 2, trajectories: 1000)
    end = .now
    let xHOPS = rhoHOPS.map { 2 * $0[0, 1].real / $0.trace.real }
    let yHOPS = rhoHOPS.map { -2 * $0[0, 1].imaginary / $0.trace.real }
    let zHOPS = rhoHOPS.map { ($0[0, 0] - $0[1, 1]) / $0.trace.real }.real
    let HOPSTime = end - start
    
    start = .now
    let (tSpaceNMQSD, rhoNMQSD) = solveModelWithNMQSD(endTime: endTime, omegaX: omegaX, g: g, omegaC: omegaC, gammaMinus: gammaMinus, trajectories: 2000)
    end = .now
    let xNMQSD = rhoNMQSD.map { 2 * $0[0, 1].real }
    let yNMQSD = rhoNMQSD.map { -2 * $0[0, 1].imaginary }
    let zNMQSD = rhoNMQSD.map { $0[0, 0] - $0[1, 1] }.real
    let NMQSDTime = end - start
    
    print("Master equation simulation time:", METime)
    print("HOPS simulation time:", HOPSTime)
    print("NMQSD simulation time:", NMQSDTime)
    
    plt.figure()
    plt.plot(x: tSpaceME, y: xME, label: "<x>_ME")
    plt.plot(x: tSpaceME, y: yME, label: "<y>_ME")
    plt.plot(x: tSpaceME, y: zME, label: "<z>_ME")
//    
    plt.plot(x: tSpaceHOPS, y: xHOPS, label: "<x>_HOPS", linestyle: "--")
    plt.plot(x: tSpaceHOPS, y: yHOPS, label: "<y>_HOPS", linestyle: "--")
    plt.plot(x: tSpaceHOPS, y: zHOPS, label: "<z>_HOPS", linestyle: "--")
    
//    plt.plot(x: tSpaceNMQSD, y: xNMQSD, label: "<x>_NMQSD", linestyle: "--")
//    plt.plot(x: tSpaceNMQSD, y: yNMQSD, label: "<y>_NMQSD", linestyle: "--")
//    plt.plot(x: tSpaceNMQSD, y: zNMQSD, label: "<z>_NMQSD", linestyle: "--")
    
    plt.xlabel("t")
    plt.ylabel("<o>")
    plt.legend()
    plt.show()
    plt.close()
    
}

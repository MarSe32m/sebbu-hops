//
//  NonLinearMultiParticleHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.10.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

public extension HOPSMultiParticleHierarchy {
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: [Noise], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The time dependent Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: [Noise], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        precondition(z.count == L.count, "The number of noise processes must equal the number of coupling operators.")
        let dimension = initialState.count
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialStateVector[i] = initialState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        var systemState: Vector<Complex<Double>> = initialState
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.map { $0.conjugateTranspose }
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialStateVector.count), .zero(G.count)], count: 4)
        var LDaggerExpectations: Vector<Complex<Double>> = .zero(LDagger.count)
        var noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: LDagger.count)
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = RK45FixedStep<[Vector<Complex<Double>>]>(initialState: [initialStateVector, initialStateVectorForShift], t0: start, dt: stepSize) { t, currentStates in
                let currentState = currentStates[0]
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                let normSquared = systemState.normSquared
                for i in 0..<LDaggerExpectations.count {
                    LDaggerExpectations[i] = systemState.inner(systemState, metric: LDagger[i]) / normSquared
                }
                
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                
                // Noise shift
                let xi = currentStates[1]
                
                M.dot(LDaggerExpectations, into: &result[1])
                for i in result[1].components.indices {
                    result[1][i] = Relaxed.multiplyAdd(WConjugateVector[i], xi[i], result[1][i])
                }
                for i in shiftIndices.indices {
                    noiseShifts[i] = .zero
                    for j in shiftIndices[i] {
                        noiseShifts[i] = Relaxed.sum(noiseShifts[i], xi[j])
                    }
                }
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                for i in L.indices {
                    Heff.add(L[i], multiplied: z[i](t).conjugate + noiseShifts[i])
                    if shiftType == .meanField {
                        Heff.add(LDagger[i], multiplied: -noiseShifts[i].conjugate)
                    }
                }
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                let kWSpan = kWArray.span
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        var kWIndex = 0
                        while index < resultBuffer.count {
                            Heff._dot(currentStatePointer, into: resultPointer)
                            let kW = kWSpan[unchecked: kWIndex]
                            for i in 0..<dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                            }
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                            kWIndex &+= 1
                        }
                    }
                }
                B.dot(currentState, addingInto: &result[0])
                for i in P.indices {
                    P[i].dot(currentState, multiplied: LDaggerExpectations[i], addingInto: &result[0])
                }
                if shiftType == .meanField {
                    for i in N.indices {
                        N[i].dot(currentState, multiplied: -LDaggerExpectations[i].conjugate, addingInto: &result[0])
                    }
                }
                return result
            }
            let resultDimension = includeHierarchy ? initialStateVector.count : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(tSpace.capacity)
            for _ in 0..<trajectory.capacity { trajectory.append(.zero(resultDimension)) }
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    trajectory[index][i] = state[0][i]
                }
                let shift = state[1].components.reduce(.zero, +)
                tSpace.append(t)
                index += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
    
    //MARK: SDE version
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoise: The white noise process
    ///   - diffusionOperator: The diffusion operator corresponding to the white noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: [Noise], whiteNoise: WhiteNoise, diffusionOperator: Matrix<Complex<Double>>, shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, whiteNoise: whiteNoise, diffusionOperator: diffusionOperator, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The time dependent Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoise: The white noise process
    ///   - diffusionOperator: The diffusion operator corresponding to the white noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: [Noise], whiteNoise: WhiteNoise, diffusionOperator: Matrix<Complex<Double>>, shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(z.count == L.count, "The number of noise processes must equal the number of coupling operators.")
        let dimension = initialState.count
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialStateVector[i] = initialState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        var systemState: Vector<Complex<Double>> = initialState
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.map { $0.conjugateTranspose }
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialStateVector.count), .zero(G.count)], count: 4)
        var LDaggerExpectations: Vector<Complex<Double>> = .zero(LDagger.count)
        var noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: LDagger.count)
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = SRK2FixedStep<[Vector<Complex<Double>>], Complex<Double>>(initialState: [initialStateVector, initialStateVectorForShift], t0: start, dt: stepSize) { t, currentStates in
                let currentState = currentStates[0]
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                let normSquared = systemState.normSquared
                for i in 0..<LDaggerExpectations.count {
                    LDaggerExpectations[i] = systemState.inner(systemState, metric: LDagger[i]) / normSquared
                }
                
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                
                // Noise shift
                let xi = currentStates[1]
                
                M.dot(LDaggerExpectations, into: &result[1])
                for i in result[1].components.indices {
                    result[1][i] = Relaxed.multiplyAdd(WConjugateVector[i], xi[i], result[1][i])
                }
                for i in shiftIndices.indices {
                    noiseShifts[i] = .zero
                    for j in shiftIndices[i] {
                        noiseShifts[i] = Relaxed.sum(noiseShifts[i], xi[j])
                    }
                }
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                for i in L.indices {
                    Heff.add(L[i], multiplied: z[i](t).conjugate + noiseShifts[i])
                    if shiftType == .meanField {
                        Heff.add(LDagger[i], multiplied: -noiseShifts[i].conjugate)
                    }
                }
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                let kWSpan = kWArray.span
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        var kWIndex = 0
                        while index < resultBuffer.count {
                            Heff._dot(currentStatePointer, into: resultPointer)
                            let kW = kWSpan[unchecked: kWIndex]
                            for i in 0..<dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                            }
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                            kWIndex &+= 1
                        }
                    }
                }
                B.dot(currentState, addingInto: &result[0])
                for i in P.indices {
                    P[i].dot(currentState, multiplied: LDaggerExpectations[i], addingInto: &result[0])
                }
                if shiftType == .meanField {
                    for i in N.indices {
                        N[i].dot(currentState, multiplied: -LDaggerExpectations[i].conjugate, addingInto: &result[0])
                    }
                }
                return result
            } g: { t, currentStates in
                var result = resultCache.removeFirst()
                let currentState = currentStates[0]
                defer { resultCache.append(result) }
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        while index < resultBuffer.count {
                            diffusionOperator.dot(currentStatePointer, into: resultPointer)
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                        }
                        
                    }
                }
                result[1].zeroComponents()
                return result
            } w: { t in
                whiteNoise(t)
            }
            let resultDimension = includeHierarchy ? initialStateVector.count : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tSpace.capacity)
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    trajectory[index][i] = state[0][i]
                }
                tSpace.append(t)
                index += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
    
    //MARK: SDE multiple noise version
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoises: The white noise processes
    ///   - diffusionOperators: The diffusion operators corresponding to the white noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: [Noise], whiteNoises: [WhiteNoise], diffusionOperators: [Matrix<Complex<Double>>], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, whiteNoises: whiteNoises, diffusionOperators: diffusionOperators.map { O in { (_: Double, _: Vector<Complex<Double>>) in O } }, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoises: The white noise processes
    ///   - diffusionOperators: The diffusion operators corresponding to the white noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: [Noise], whiteNoises: [WhiteNoise], diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, whiteNoises: whiteNoises, diffusionOperators: diffusionOperators, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The time dependent Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoises: The white noise processes
    ///   - diffusionOperators: The diffusion operators corresponding to the white noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: [Noise], whiteNoises: [WhiteNoise], diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(z.count == L.count, "The number of noise processes must equal the number of coupling operators.")
        let dimension = initialState.count
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialStateVector[i] = initialState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        var systemState: Vector<Complex<Double>> = initialState
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.map { $0.conjugateTranspose }
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialStateVector.count), .zero(G.count)], count: 4)
        var LDaggerExpectations: Vector<Complex<Double>> = .zero(LDagger.count)
        var noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: LDagger.count)
        
        // Convert diffusionOperators into the form that the SRK2 solver expects
        let g = diffusionOperators.map { g in
            //TODO: Since the SRK2 solver copies these results into a cache vectors, can we somehow reuse this result vector?
            var result: [Vector<Complex<Double>>] = [.zero(initialStateVector.count), .zero(G.count)]
            var systemState: Vector<Complex<Double>> = .zero(dimension)
            return { (_ t: Double , _ currentStates: [Vector<Complex<Double>>]) in
                for i in 0..<dimension {
                    systemState[i] = currentStates[0][i]
                }
                let O = g(t, systemState)
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentStates[0].components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        while index < resultBuffer.count {
                            O.dot(currentStatePointer, into: resultPointer)
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                        }
                        
                    }
                }
                // We don't need to zero the components for the shift here, since we never modify them
                //result[1].zeroComponents()
                return result
            }
        }
        
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = SRK2FixedStepMultiNoise(initialState: [initialStateVector, initialStateVectorForShift], t0: start, dt: stepSize, f: { t, currentStates in
                let currentState = currentStates[0]
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                let normSquared = systemState.normSquared
                for i in 0..<LDaggerExpectations.count {
                    LDaggerExpectations[i] = systemState.inner(systemState, metric: LDagger[i]) / normSquared
                }
                
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                
                // Noise shift
                let xi = currentStates[1]
                
                M.dot(LDaggerExpectations, into: &result[1])
                for i in result[1].components.indices {
                    result[1][i] = Relaxed.multiplyAdd(WConjugateVector[i], xi[i], result[1][i])
                }
                for i in shiftIndices.indices {
                    noiseShifts[i] = .zero
                    for j in shiftIndices[i] {
                        noiseShifts[i] = Relaxed.sum(noiseShifts[i], xi[j])
                    }
                }
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                for i in L.indices {
                    Heff.add(L[i], multiplied: z[i](t).conjugate + noiseShifts[i])
                    if shiftType == .meanField {
                        Heff.add(LDagger[i], multiplied: -noiseShifts[i].conjugate)
                    }
                }
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                let kWSpan = kWArray.span
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        var kWIndex = 0
                        while index < resultBuffer.count {
                            Heff._dot(currentStatePointer, into: resultPointer)
                            let kW = kWSpan[unchecked: kWIndex]
                            for i in 0..<dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                            }
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                            kWIndex &+= 1
                        }
                    }
                }
                B.dot(currentState, addingInto: &result[0])
                for i in P.indices {
                    P[i].dot(currentState, multiplied: LDaggerExpectations[i], addingInto: &result[0])
                }
                if shiftType == .meanField {
                    for i in N.indices {
                        N[i].dot(currentState, multiplied: -LDaggerExpectations[i].conjugate, addingInto: &result[0])
                    }
                }
                return result
            }, g: g, w: whiteNoises.map { w in { (t: Double) in w(t)}})
            let resultDimension = includeHierarchy ? initialStateVector.count : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tSpace.capacity)
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    trajectory[index][i] = state[0][i]
                }
                tSpace.append(t)
                index += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
}



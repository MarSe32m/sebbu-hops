//
//  NonLinearHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

public extension HOPSHierarchy {
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
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, whiteNoise: WhiteNoise, diffusionOperator: Matrix<Complex<Double>>, shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
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
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, whiteNoise: WhiteNoise, diffusionOperator: Matrix<Complex<Double>>, shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        let dimension = initialState.count
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialStateVector[i] = initialState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        var systemState: Vector<Complex<Double>> = initialState
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.conjugateTranspose
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialStateVector.count), .zero(G.count)], count: 4)
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = SRK2FixedStep<[Vector<Complex<Double>>], Complex<Double>>(initialState: [initialStateVector, initialStateVectorForShift], t0: start, dt: stepSize) { t, currentStates in
                let currentState = currentStates[0]
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                let LExpectation = systemState.inner(systemState, metric: L) / systemState.normSquared
                let LDaggerExpectation = LExpectation.conjugate
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                
                // Noise shift
                let xi = currentStates[1]
                var shift: Complex<Double> = .zero
            
                result[1].components.withUnsafeMutableBufferPointer { result in
                    for i in result.indices {
                        result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                        shift = Relaxed.sum(shift, xi[i])
                    }
                }
                let zTilde = z(t).conjugate + shift
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                Heff.add(L, multiplied: zTilde)
                if shiftType == .meanField {
                    Heff.add(LDagger, multiplied: -shift.conjugate)
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
                            Heff.dot(currentStatePointer, into: resultPointer)
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
                P.dot(currentState, multiplied: LDaggerExpectation, addingInto: &result[0])
                if shiftType == .meanField {
                    N.dot(currentState, multiplied: -LExpectation, addingInto: &result[0])
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
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, whiteNoises: [WhiteNoise], diffusionOperators: [Matrix<Complex<Double>>], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
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
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, whiteNoises: [WhiteNoise], diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
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
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, whiteNoises: [WhiteNoise], diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        let dimension = initialState.count
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialStateVector[i] = initialState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        var systemState: Vector<Complex<Double>> = initialState
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.conjugateTranspose
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialStateVector.count), .zero(G.count)], count: 4)
        
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
                let LExpectation = systemState.inner(systemState, metric: L) / systemState.normSquared
                let LDaggerExpectation = LExpectation.conjugate
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                
                // Noise shift
                let xi = currentStates[1]
                var shift: Complex<Double> = .zero
            
                result[1].components.withUnsafeMutableBufferPointer { result in
                    for i in result.indices {
                        result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                        shift = Relaxed.sum(shift, xi[i])
                    }
                }
                let zTilde = z(t).conjugate + shift
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                Heff.add(L, multiplied: zTilde)
                if shiftType == .meanField {
                    Heff.add(LDagger, multiplied: -shift.conjugate)
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
                            Heff.dot(currentStatePointer, into: resultPointer)
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
                P.dot(currentState, multiplied: LDaggerExpectation, addingInto: &result[0])
                if shiftType == .meanField {
                    N.dot(currentState, multiplied: -LExpectation, addingInto: &result[0])
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



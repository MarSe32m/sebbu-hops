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
import Algorithms

public extension HOPSHierarchy {
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
    func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, customOperators: customOperators, stepSize: stepSize)
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
    @inline(__always)
    func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
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
            var solver = RK45FixedStep<[Vector<Complex<Double>>]>(initialState: [initialStateVector, initialStateVectorForShift], t0: start, dt: stepSize) { t, currentStates in
                let currentState = currentStates[0]
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                let LDaggerExpectation = systemState.inner(systemState, metric: LDagger) / systemState.normSquared
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                
                // Noise shift
                let xi = currentStates[1]
                var shift: Complex<Double> = .zero
            
                result[1].components.withUnsafeMutableBufferPointer { result in
                    var i = 0
                    while i &+ 4 <= result.count {
                        result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                        result[i &+ 1] = Relaxed.multiplyAdd(GConjugateVector[i &+ 1], LDaggerExpectation, Relaxed.product(WConjugateVector[i &+ 1], xi[i &+ 1]))
                        result[i &+ 2] = Relaxed.multiplyAdd(GConjugateVector[i &+ 2], LDaggerExpectation, Relaxed.product(WConjugateVector[i &+ 2], xi[i &+ 2]))
                        result[i &+ 3] = Relaxed.multiplyAdd(GConjugateVector[i &+ 3], LDaggerExpectation, Relaxed.product(WConjugateVector[i &+ 3], xi[i &+ 3]))
                        shift = Relaxed.sum(shift, xi[i])
                        shift = Relaxed.sum(shift, xi[i &+ 1])
                        shift = Relaxed.sum(shift, xi[i &+ 2])
                        shift = Relaxed.sum(shift, xi[i &+ 3])
                        i &+= 4
                    }
                    while i < result.count {
                        result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                        shift = Relaxed.sum(shift, xi[i])
                        i &+= 1
                    }
                }
                let zTilde = z(t).conjugate + shift
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                Heff.add(L, multiplied: zTilde)
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        var kWIndex = 0
                        while index < resultBuffer.count {
                            Heff.dot(currentStatePointer, into: resultPointer)
                            let kW = kWArray[kWIndex]
                            var i = 0
                            while i &+ 2 <= dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                resultPointer[i &+ 1] = Relaxed.multiplyAdd(kW, currentStatePointer[i &+ 1], resultPointer[i &+ 1])
                                i &+= 2
                            }
                            while i < dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                i &+= 1
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
                return result
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(dimension), count: tSpace.capacity)
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(dimension))
                }
                for i in 0..<dimension {
                    trajectory[index][i] = state[0][i]
                }
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
    ///   - w: The white noise process
    ///   - wOperator: The operator corresponding to the white noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, w: WhiteNoise, wOperator: Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, w: w, wOperator: wOperator, customOperators: customOperators, stepSize: stepSize)
    }
    
    /// Solve the non-linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The time dependent Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - w: The white noise process
    ///   - wOperator: The operator corresponding to the white noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding **unnormalized** system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, w: WhiteNoise, wOperator: Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
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
                let LDaggerExpectation = systemState.inner(systemState, metric: LDagger) / systemState.normSquared
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                
                // Noise shift
                let xi = currentStates[1]
                var shift: Complex<Double> = .zero
            
                result[1].components.withUnsafeMutableBufferPointer { result in
                    var i = 0
                    while i &+ 4 <= result.count {
                        result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                        result[i &+ 1] = Relaxed.multiplyAdd(GConjugateVector[i &+ 1], LDaggerExpectation, Relaxed.product(WConjugateVector[i &+ 1], xi[i &+ 1]))
                        result[i &+ 2] = Relaxed.multiplyAdd(GConjugateVector[i &+ 2], LDaggerExpectation, Relaxed.product(WConjugateVector[i &+ 2], xi[i &+ 2]))
                        result[i &+ 3] = Relaxed.multiplyAdd(GConjugateVector[i &+ 3], LDaggerExpectation, Relaxed.product(WConjugateVector[i &+ 3], xi[i &+ 3]))
                        shift = Relaxed.sum(shift, xi[i])
                        shift = Relaxed.sum(shift, xi[i &+ 1])
                        shift = Relaxed.sum(shift, xi[i &+ 2])
                        shift = Relaxed.sum(shift, xi[i &+ 3])
                        i &+= 4
                    }
                    while i < result.count {
                        result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                        shift = Relaxed.sum(shift, xi[i])
                        i &+= 1
                    }
                }
                let zTilde = z(t).conjugate + shift
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                Heff.add(L, multiplied: zTilde)
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                result[0].zeroComponents()
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState.components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        var kWIndex = 0
                        while index < resultBuffer.count {
                            Heff.dot(currentStatePointer, into: resultPointer)
                            let kW = kWArray[kWIndex]
                            var i = 0
                            while i &+ 2 <= dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                resultPointer[i &+ 1] = Relaxed.multiplyAdd(kW, currentStatePointer[i &+ 1], resultPointer[i &+ 1])
                                i &+= 2
                            }
                            while i < dimension {
                                resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                                i &+= 1
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
                            wOperator.dot(currentStatePointer, into: resultPointer)
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                        }
                        
                    }
                }
                return result
            } w: { t in
                w(t)
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(dimension), count: tSpace.capacity)
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(dimension))
                }
                for i in 0..<dimension {
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



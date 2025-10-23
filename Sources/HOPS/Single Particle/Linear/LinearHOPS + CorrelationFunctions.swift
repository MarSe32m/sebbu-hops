//
//  LinearHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

//MARK: Two-time correlation functions
public extension HOPSHierarchy {
    /// Calculate two-time correlatior <A(t)B(s)> for a noise realization.
    /// - Parameters:
    ///   - start: Start time of the evolution.
    ///   - t: Time at which the operator A is applied.
    ///   - A: Operator A in the two-time correlator.
    ///   - s: Time at which the operator B is applied.
    ///   - B: Operator B in the two-time correlator.
    ///   - initialState: Initial system state.
    ///   - H: Hamiltonian of the system.
    ///   - z: Noise process of the environment.
    ///   - customOperators: Custom operators.
    ///   - stepSize: Step size used in the propagation. Default is 0.01.
    ///   - includeHierarchy: Whether to include the whole hierarchy in the returned states. Default false.
    /// - Returns: A tuple (tauSpace, braTrajectory, ketTrajectory, correlationFunction) where tauSpace is [0, max(t,s)], braTrajectory contains the bra state evolution, ketTrajectory contains the ket state evolution and correlationFunction contains the two-time correlator <A(t)B(s)> at (t, s).
    @inlinable
    @inline(__always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(start: Double = 0.0,
                                                t: Double, A: Matrix<Complex<Double>>,
                                                s: Double, B: Matrix<Complex<Double>>,
                                                initialState: Vector<Complex<Double>>,
                                                H: Matrix<Complex<Double>>,
                                                z: Noise,
                                                customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                                stepSize: Double = 0.01,
                                                includeHierarchy: Bool = false) -> (tauSpace: [Double], braTrajectory: [Vector<Complex<Double>>], ketTrajectory: [Vector<Complex<Double>>], correlationFunction: Complex<Double>) where Noise: ComplexNoiseProcess {
        solveLinearTwoTimeCorrelationFunction(start: start, t: t, A: A, s: s, B: B, initialState: initialState, H: { _ in H }, z: z, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Calculate two-time correlator <A(t)B(s)> for a noise realization.
    /// - Parameters:
    ///   - start: Start time of the evolution
    ///   - t: Time at which the operator A is applied.
    ///   - A: Operator A in the two-time correlator.
    ///   - s: Time at which the operator B is applied.
    ///   - B: Operator B in the two-time correlator.
    ///   - initialState: Initial system state.
    ///   - H: Hamiltonian of the system
    ///   - z: Nosie process of the environment.
    ///   - customOperators: Custom operators.
    ///   - stepSize: Step size used in the propagation. Default is 0.01.
    ///   - includeHierarchy: Whether to include the whole hierarchy in the resturned states. Default is false.
    /// - Returns: A tuple (tauSpace, braTrajectory, ketTrajectory, correlationFunction) where tauSpace is [0,max(t,s)], braTrajectory contains the braState evolution, ketTrajectory contains the ket state evolution and correlationFunction contains the two-time correlator <A(t)B(s)> at (t, s).
    @inlinable
    func solveLinearTwoTimeCorrelationFunction<Noise>(start: Double = 0.0,
                                                t: Double, A: Matrix<Complex<Double>>,
                                                s: Double, B: Matrix<Complex<Double>>,
                                                initialState: Vector<Complex<Double>>,
                                                H: (Double) -> Matrix<Complex<Double>>,
                                                z: Noise,
                                                customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                                stepSize: Double = 0.01,
                                                includeHierarchy: Bool = false) -> (tauSpace: [Double], braTrajectory: [Vector<Complex<Double>>], ketTrajectory: [Vector<Complex<Double>>], correlationFunction: Complex<Double>) where Noise: ComplexNoiseProcess {
        if t == s {
            let (tauSpace, trajectory) = solveLinear(start: start, end: t, initialState: initialState, H: H, z: z, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
            let braTrajectory = trajectory
            let ketTrajectory = trajectory
            let state = trajectory.last!
            let correlationFunction = state.inner(state, metric: A.dot(B))
            return (tauSpace, braTrajectory, ketTrajectory, correlationFunction)
        }
        let perturbationTime = min(t, s)
        let end = max(t, s)
        
        return withoutActuallyEscaping(H) { H in
            var propagator = linearDyadicPropagator(start: start, initialSystemState: initialState, H: H, z: z, customOperators: customOperators, stepSize: stepSize)
            let resultDimension = includeHierarchy ? self.B.columns : dimension
            var tauSpace: [Double] = []
            tauSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var braTrajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tauSpace.capacity)
            var ketTrajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tauSpace.capacity)
            var index = 0
            while propagator.t < perturbationTime {
                let (tau, state) = propagator.step()
                if index >= braTrajectory.count {
                    braTrajectory.append(.zero(resultDimension))
                }
                if index >= ketTrajectory.count {
                    ketTrajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    braTrajectory[index][i] = state[0][i]
                    ketTrajectory[index][i] = state[1][i]
                }
                tauSpace.append(tau)
                index += 1
            }
            // Apply perturbation operator
            let O = t > s ? B : A.conjugateTranspose
            let stateIndex = t > s ? 1 : 0
            var nextInitialState: [Vector<Complex<Double>>] = propagator.currentState
            propagator.currentState[stateIndex].components.withUnsafeBufferPointer { lastStateComponents in
                nextInitialState[stateIndex].components.withUnsafeMutableBufferPointer { nextInitialStateComponents in
                    var idx = 0
                    var lastStateComponentPointer = lastStateComponents.baseAddress!
                    var nextInitialStateComponentPointer = nextInitialStateComponents.baseAddress!
                    while idx < lastStateComponents.count {
                        O.dot(lastStateComponentPointer, into: nextInitialStateComponentPointer)
                        lastStateComponentPointer += dimension
                        nextInitialStateComponentPointer += dimension
                        idx += dimension
                    }
                }
            }
            if index > 0 {
                index -= 1
                tauSpace.removeLast()
            }
            propagator.reset(initialState: nextInitialState, t0: propagator.t)
            while propagator.t < end {
                let (tau, state) = propagator.step()
                if index >= braTrajectory.count {
                    braTrajectory.append(.zero(resultDimension))
                }
                if index >= ketTrajectory.count {
                    ketTrajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    braTrajectory[index][i] = state[0][i]
                    ketTrajectory[index][i] = state[1][i]
                }
                tauSpace.append(tau)
                index += 1
            }
            
            braTrajectory.removeLast(braTrajectory.count - tauSpace.count)
            ketTrajectory.removeLast(ketTrajectory.count - tauSpace.count)
            let braState = braTrajectory.last!
            let ketState = ketTrajectory.last!
            let correlationFunction = braState.inner(ketState, metric: t > s ? A : B)
            return (tauSpace, braTrajectory, ketTrajectory, correlationFunction)
        }
    }
    
    @inlinable
    func linearDyadicPropagator<Noise>(start: Double, initialSystemState: Vector<Complex<Double>>, H: @escaping (Double) -> Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess {
        precondition(initialSystemState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialStateVector: Vector<Complex<Double>> = .zero(self.B.columns)
        for i in 0..<initialSystemState.count {
            initialStateVector[i] = initialSystemState[i]
        }
        return linearDyadicPropagator(start: start, initialBraHierarchyState: initialStateVector, initialKetHierarchyState: initialStateVector, H: H, z: z, customOperators: customOperators, stepSize: stepSize)
    }
    
    @inlinable
    func linearDyadicPropagator<Noise>(start: Double, initialBraHierarchyState: Vector<Complex<Double>>, initialKetHierarchyState: Vector<Complex<Double>>, H: @escaping (Double) -> Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess {
        var braState: Vector<Complex<Double>> = .zero(dimension)
        var ketState: Vector<Complex<Double>> = .zero(dimension)
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialBraHierarchyState.count), .zero(initialKetHierarchyState.count)], count: 4)
        var Heff = H(start)
        let solver = RK45FixedStep(initialState: [initialBraHierarchyState, initialKetHierarchyState], t0: start, dt: stepSize) { t, currentState in
            for i in 0..<dimension {
                braState[i] = currentState[0][i]
                ketState[i] = currentState[1][i]
            }
            var result = resultCache.removeFirst()
            defer { resultCache.append(result) }
            // Bra propagation
            Heff.zeroElements()
            Heff.add(H(t), multiplied: -.i)
            let z = z(t).conjugate
            Heff.add(L, multiplied: z)
            for customOperator in customOperators {
                Heff.add(customOperator(t, braState))
            }
            let kWSpan = kWArray.span
            result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                currentState[0].components.withUnsafeBufferPointer { currentStateBuffer in
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
            self.B.dot(currentState[0], addingInto: &result[0])
            
            // Ket propagation
            Heff.zeroElements()
            Heff.add(H(t), multiplied: -.i)
            Heff.add(L, multiplied: z)
            for customOperator in customOperators {
                Heff.add(customOperator(t, ketState))
            }
            result[1].components.withUnsafeMutableBufferPointer { resultBuffer in
                currentState[1].components.withUnsafeBufferPointer { currentStateBuffer in
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
            self.B.dot(currentState[1], addingInto: &result[1])
            return result
        }
        return HOPSPropagator(solver)
    }
}

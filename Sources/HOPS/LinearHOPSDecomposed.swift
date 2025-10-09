//
//  LinearHOPSDecomposed.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 7.10.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

public extension HOPSHierarchy {
    /// Solve the linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding system state vectors
    @inlinable
    @inline(__always)
    func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, O: Matrix<Complex<Double>>, at: Double, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, O: O, at: at, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The time dependent Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding system state vectors
    @inlinable
    func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, O: Matrix<Complex<Double>>, at: Double, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<initialState.count {
            initialStateVector[i] = initialState[i]
        }
        var systemState = initialState
        var resultCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(initialStateVector.count), count: 4)
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = RK45FixedStep(initialState: initialStateVector, t0: start, dt: stepSize) { t, currentState in
                for i in 0..<dimension {
                    systemState[i] = currentState[i]
                }
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                let z = z(t).conjugate
                Heff.add(L, multiplied: z)
                for customOperator in customOperators {
                    Heff.add(customOperator(t, systemState))
                }
                let kWSpan = kWArray.span
                result.components.withUnsafeMutableBufferPointer { resultBuffer in
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
                B.dot(currentState, addingInto: &result)
                return result
            }
            let resultDimension = includeHierarchy ? initialStateVector.count : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(tSpace.capacity)
            for _ in 0..<trajectory.capacity { trajectory.append(.zero(resultDimension)) }
            var index = 0
            while solver.t < min(end, at) {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    trajectory[index][i] = state[i]
                }
                tSpace.append(t)
                index += 1
            }
            
            let (s, lastState) = solver.step()
            var nextInitialState: Vector<Complex<Double>> = .zero(lastState.count)
            nextInitialState.copyComponents(from: lastState)
            lastState.components.withUnsafeBufferPointer { lastStateComponents in
                nextInitialState.components.withUnsafeMutableBufferPointer { nextInitialStateComponents in
                    var index = 0
                    var lastStateComponentPointer = lastStateComponents.baseAddress!
                    var nextInitialStateComponentPointer = nextInitialStateComponents.baseAddress!
                    while index < lastStateComponents.count {
                        O.dot(lastStateComponentPointer, into: nextInitialStateComponentPointer)
                        lastStateComponentPointer += dimension
                        nextInitialStateComponentPointer += dimension
                        index += dimension
                    }
                }
            }
            solver.reset(initialState: nextInitialState, t0: s)
            
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    trajectory[index][i] = state[i]
                }
                tSpace.append(t)
                index += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }
}

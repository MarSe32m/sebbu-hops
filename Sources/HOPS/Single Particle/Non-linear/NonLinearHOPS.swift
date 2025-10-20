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
    func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
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
    func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, shiftType: ShiftType = .none, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        return withoutActuallyEscaping(H) { H in
            var propagator = nonLinearPropagator(start: start, initialSystemState: initialState, H: H, z: z, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize)
            let resultDimension = includeHierarchy ? B.columns : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            //var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tSpace.capacity)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(tSpace.capacity)
            for _ in 0..<trajectory.capacity { trajectory.append(.zero(resultDimension)) }
            var index = 0
            while propagator.t < end {
                let (t, state) = propagator.step()
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
    
    @inlinable
    func nonLinearPropagator<Noise>(start: Double, initialSystemState: Vector<Complex<Double>>, H: @escaping (Double) -> Matrix<Complex<Double>>, z: Noise, shiftType: ShiftType, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess {
        precondition(initialSystemState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<initialSystemState.count {
            initialStateVector[i] = initialSystemState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        return nonLinearPropagator(start: start, initialHierarchyState: initialStateVector, initialShiftVector: initialStateVectorForShift, H: H, z: z, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize)
    }
    
    @inlinable
    func nonLinearPropagator<Noise>(start: Double, initialHierarchyState: Vector<Complex<Double>>, initialShiftVector: Vector<Complex<Double>>, H: @escaping (Double) -> Matrix<Complex<Double>>, z: Noise, shiftType: ShiftType, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess {
        var systemState: Vector<Complex<Double>> = .zero(dimension)
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.conjugateTranspose
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialHierarchyState.count), .zero(initialShiftVector.count)], count: 4)
        var Heff = H(start)
        let solver = RK45FixedStep<[Vector<Complex<Double>>]>(initialState: [initialHierarchyState, initialShiftVector], t0: start, dt: stepSize) { t, currentStates in
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
            P.dot(currentState, multiplied: LDaggerExpectation, addingInto: &result[0])
            if shiftType == .meanField {
                N.dot(currentState, multiplied: -LExpectation, addingInto: &result[0])
            }
            return result
        }
        return HOPSPropagator(solver)
    }
}



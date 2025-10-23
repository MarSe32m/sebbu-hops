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
    @inlinable
    @inline(__always)
    func solveNonLinearTwoTimeCorrelationFunction<Noise>(start: Double = 0.0,
                                                t: Double, A: Matrix<Complex<Double>>,
                                                s: Double, B: Matrix<Complex<Double>>,
                                                initialState: Vector<Complex<Double>>,
                                                H: Matrix<Complex<Double>>,
                                                z: Noise,
                                                shiftType: ShiftType = .none,
                                                customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                                stepSize: Double = 0.01,
                                                includeHierarchy: Bool = false) -> (tauSpace: [Double], braTrajectory: [Vector<Complex<Double>>], ketTrajectory: [Vector<Complex<Double>>], correlationFunction: Complex<Double>, normalizationFactor: [Double]) where Noise: ComplexNoiseProcess {
        solveNonLinearTwoTimeCorrelationFunction(start: start, t: t, A: A, s: s, B: B, initialState: initialState, H: { _ in H }, z: z, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solveNonLinearTwoTimeCorrelationFunction<Noise>(start: Double = 0.0,
                                                t: Double, A: Matrix<Complex<Double>>,
                                                s: Double, B: Matrix<Complex<Double>>,
                                                initialState: Vector<Complex<Double>>,
                                                H: (Double) -> Matrix<Complex<Double>>,
                                                z: Noise,
                                                shiftType: ShiftType,
                                                customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                                stepSize: Double = 0.01,
                                                         includeHierarchy: Bool = false) -> (tauSpace: [Double], braTrajectory: [Vector<Complex<Double>>], ketTrajectory: [Vector<Complex<Double>>], correlationFunction: Complex<Double>, normalizationFactor: [Double]) where Noise: ComplexNoiseProcess {
        if t == s {
            let (tauSpace, trajectory) = solveNonLinear(start: start, end: t, initialState: initialState, H: H, z: z, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
            let braTrajectory = trajectory
            let ketTrajectory = trajectory
            let normalizationFactor = braTrajectory.map { $0.normSquared }
            let state = Vector(Array(trajectory.last!.components[0..<dimension]))
            let correlationFunction = state.inner(state, metric: A.dot(B)) / state.normSquared
            return (tauSpace, braTrajectory, ketTrajectory, correlationFunction, normalizationFactor)
        }
        let perturbationTime = min(t, s)
        let end = max(t, s)
        
        return withoutActuallyEscaping(H) { H in
            var propagator = nonLinearDyadicPropagator(start: start, normalizeByBra: t > s, initialSystemState: initialState, H: H, z: z, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize)
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
            let normalizationFactor = zip(braTrajectory, ketTrajectory).map { bra, ket in
                t > s ? Vector(Array(bra.components[0..<dimension])).normSquared : Vector(Array(ket.components[0..<dimension])).normSquared
            }
            return (tauSpace, braTrajectory, ketTrajectory, correlationFunction, normalizationFactor)
        }
    }
    
    @inlinable
    func nonLinearDyadicPropagator<Noise>(start: Double, normalizeByBra: Bool, initialSystemState: Vector<Complex<Double>>, H: @escaping (Double) -> Matrix<Complex<Double>>, z: Noise, shiftType: ShiftType, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess {
        precondition(initialSystemState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<initialSystemState.count {
            initialStateVector[i] = initialSystemState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        return nonLinearDyadicPropagator(start: start, normalizeByBra: normalizeByBra, initialBraHierarchyState: initialStateVector, initialKetHierarchyState: initialStateVector, initialShiftVector: initialStateVectorForShift, H: H, z: z, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize)
    }
    
    @inlinable
    func nonLinearDyadicPropagator<Noise>(start: Double, normalizeByBra: Bool, initialBraHierarchyState: Vector<Complex<Double>>, initialKetHierarchyState: Vector<Complex<Double>>, initialShiftVector: Vector<Complex<Double>>, H: @escaping (Double) -> Matrix<Complex<Double>>, z: Noise, shiftType: ShiftType, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>], stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess {
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.conjugateTranspose
        
        var braState: Vector<Complex<Double>> = .zero(dimension)
        var ketState: Vector<Complex<Double>> = .zero(dimension)
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialBraHierarchyState.count), .zero(initialKetHierarchyState.count), .zero(initialShiftVector.count)], count: 4)
        var Heff = H(start)
        let solver = RK45FixedStep<[Vector<Complex<Double>>]>(initialState: [initialBraHierarchyState, initialKetHierarchyState, initialShiftVector], t0: start, dt: stepSize) { t, currentStates in
            for i in 0..<dimension {
                braState[i] = currentStates[0][i]
                ketState[i] = currentStates[1][i]
            }
            let LExpectation = normalizeByBra ? (braState.inner(braState, metric: L) / braState.normSquared) : (ketState.inner(ketState, metric: L) / ketState.normSquared)
            let LDaggerExpectation = LExpectation.conjugate
            
            var result = resultCache.removeFirst()
            defer { resultCache.append(result) }
            
            // Noise shift
            let xi = currentStates[2]
            var shift: Complex<Double> = .zero
        
            result[2].components.withUnsafeMutableBufferPointer { result in
                for i in result.indices {
                    result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                    shift = Relaxed.sum(shift, xi[i])
                }
            }
            let zTilde = z(t).conjugate + shift
            
            // Bra propagation
            Heff.zeroElements()
            Heff.add(H(t), multiplied: -.i)
            Heff.add(L, multiplied: zTilde)
            if shiftType == .meanField {
                Heff.add(LDagger, multiplied: -shift.conjugate)
            }
            for customOperator in customOperators {
                Heff.add(customOperator(t, braState))
            }
            let kWSpan = kWArray.span
            result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                currentStates[0].components.withUnsafeBufferPointer { currentStateBuffer in
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
            B.dot(currentStates[0], addingInto: &result[0])
            P.dot(currentStates[0], multiplied: LDaggerExpectation, addingInto: &result[0])
            if shiftType == .meanField {
                N.dot(currentStates[0], multiplied: -LExpectation, addingInto: &result[0])
            }
            
            // Ket propagation
            Heff.zeroElements()
            Heff.add(H(t), multiplied: -.i)
            Heff.add(L, multiplied: zTilde)
            if shiftType == .meanField {
                Heff.add(LDagger, multiplied: -shift.conjugate)
            }
            for customOperator in customOperators {
                Heff.add(customOperator(t, ketState))
            }
            result[1].components.withUnsafeMutableBufferPointer { resultBuffer in
                currentStates[1].components.withUnsafeBufferPointer { currentStateBuffer in
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
            B.dot(currentStates[1], addingInto: &result[1])
            P.dot(currentStates[1], multiplied: LDaggerExpectation, addingInto: &result[1])
            if shiftType == .meanField {
                N.dot(currentStates[1], multiplied: -LExpectation, addingInto: &result[1])
            }
            return result
        }
        return HOPSPropagator(solver)
    }
}



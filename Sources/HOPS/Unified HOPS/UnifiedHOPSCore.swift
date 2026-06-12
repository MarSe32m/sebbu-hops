//
//  UnifiedHOPSCore.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 11.6.2026.
//

import SebbuScience
import Numerics
import BasicContainers

public extension UnifiedHOPSHierarchy {
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        equationType: EquationType,
        shiftType: ShiftType,
        customOperators: consuming Span<CustomOperator>,
        stepSize: Double,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess {
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialTotalState.count == totalDimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
            precondition(noiseShifts.count == G.count, "There must be equal amount of noise shifts as there are exponential terms in the total BCF")
            let initialState: UniqueVector<Complex<Double>> = .init(copying: initialTotalState, count: dimension)
            operation(start, initialState, initialTotalState)
            return withUnsafePointer(to: self) { hierarchy in
                let state = HOPSState(totalStateVector: initialTotalState, initialShifts: noiseShifts)
                switch equationType {
                case .linear:
                    let rhs = LinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators)
                    _solve(start: start, end: end, state: state, rhs: rhs, stepSize: stepSize, forEach: operation)
                case .nonLinear:
                    let rhs = NonLinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators)
                    _solve(start: start, end: end, state: state, rhs: rhs, stepSize: stepSize, forEach: operation)
                case .nonLinearNormalized:
                    let rhs = NonLinearNormalizedHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators)
                    _solve(start: start, end: end, state: state, rhs: rhs, stepSize: stepSize, forEach: operation)
                }
                noises = Span()
                customOperators = Span()
            }
        }
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        equationType: EquationType,
        shiftType: ShiftType,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator>,
        stepSize: Double,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialTotalState.count == totalDimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
            precondition(noiseShifts.count == G.count, "There must be equal amount of noise shifts as there are exponential terms in the total BCF")
            let initialState: UniqueVector<Complex<Double>> = .init(copying: initialTotalState, count: dimension)
            operation(start, initialState, initialTotalState)
            return withUnsafePointer(to: self) { hierarchy in
                withUnsafeTemporaryAllocation(of: Complex<Double>.self, capacity: jumpOperators.count) { noiseScratch in
                    let noiseSpan = noiseScratch.mutableSpan
                    let state = HOPSState(totalStateVector: initialTotalState, initialShifts: noiseShifts)
                    switch equationType {
                    case .linear:
                        let rhs = LinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators, jumpOperators: jumpOperators)
                        _solve(start: start, end: end, noiseSpan: noiseSpan, state: state, rhs: rhs, stepSize: stepSize, forEach: operation)
                    case .nonLinear:
                        let rhs = NonLinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators, jumpOperators: jumpOperators)
                        _solve(start: start, end: end, noiseSpan: noiseSpan, state: state, rhs: rhs, stepSize: stepSize, forEach: operation)
                    case .nonLinearNormalized:
                        let rhs = NonLinearNormalizedHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators, jumpOperators: jumpOperators)
                        _solve(start: start, end: end, noiseSpan: noiseSpan, state: state, rhs: rhs, stepSize: stepSize, forEach: operation)
                    }
                    noises = Span()
                    customOperators = Span()
                    jumpOperators = Span()
                }
            }
        }
    }
    
    @inlinable
    @inline(always)
    internal func _solve<State: ~Copyable, RHS: ~Copyable & ~Escapable>(
        start: Double,
        end: Double,
        state: consuming State,
        rhs: consuming RHS,
        stepSize: Double,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where RHS: ODERHSFunction & HOPSStateFuncProtocol, RHS.State == State, State: HOPSStateProtocol & FixedStepODESolverState {
        let k1 = rhs.zero()
        let k2 = rhs.zero()
        let k3 = rhs.zero()
        let k4 = rhs.zero()
        let temporary = rhs.zero()
        var solver = UniqueRK4Solver(t: start, dt: stepSize, rhs: rhs, k1: k1, k2: k2, k3: k3, k4: k4, temporary: temporary)
        while solver.t < end {
            let t = solver.step(y: &state)
            let systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
            operation(t, systemState, state.totalStateVector)
            let _ = systemState.consumeComponents()
        }
    }
    
    @inlinable
    @inline(always)
    internal func _solve<State: ~Copyable, RHS: ~Copyable & ~Escapable>(
        start: Double,
        end: Double,
        noiseSpan: consuming MutableSpan<Complex<Double>>,
        state: consuming State,
        rhs: consuming RHS,
        stepSize: Double,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where RHS: SDERHSFunction & HOPSStateFuncProtocol, State: HOPSStateProtocol & FixedStepSDESolverState, RHS.State == State, State.NoiseType == RHS.NoiseType, State.NoiseType == Complex<Double> {
        let drift0 = rhs.zero()
        let drift1 = rhs.zero()
        let noise0 = rhs.zero()
        let noise1 = rhs.zero()
        let temporary = rhs.zero()
        var solver = UniqueSRK2Solver(t: start, dt: stepSize, rhs: rhs, drift0: drift0, drift1: drift1, noise0: noise0, noise1: noise1, temporary: temporary, noises: noiseSpan)
        while solver.t < end {
            let t = solver.step(y: &state)
            let systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
            operation(t, systemState, state.totalStateVector)
            let _ = systemState.consumeComponents()
        }
    }
}

// MARK: Convenience methods for pure HOPS solve functions
public extension UnifiedHOPSHierarchy {
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        equationType: EquationType,
        shiftType: ShiftType,
        customOperators: consuming Span<CustomOperator>,
        stepSize: Double,
        includeHierarchy: Bool
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        var tSpace: [Double] = []
        var systemTrajectory: [Vector<Complex<Double>>] = []
        var totalTrajectory: [Vector<Complex<Double>>] = []
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize) { t, system, total in
            tSpace.append(t)
            systemTrajectory.append(.init(copying: system))
            if includeHierarchy { totalTrajectory.append(.init(copying: total)) }
        }
        return Trajectory(tSpace: tSpace, systemTrajectory: systemTrajectory, totalTrajectory: totalTrajectory)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
}

// MARK: Convenience methods for QSD+HOPS solve functions
public extension UnifiedHOPSHierarchy {
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        equationType: EquationType,
        shiftType: ShiftType,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator>,
        stepSize: Double,
        includeHierarchy: Bool
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        var tSpace: [Double] = []
        var systemTrajectory: [Vector<Complex<Double>>] = []
        var totalTrajectory: [Vector<Complex<Double>>] = []
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, equationType: equationType, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize) { t, system, total in
            tSpace.append(t)
            systemTrajectory.append(.init(copying: system))
            if includeHierarchy { totalTrajectory.append(.init(copying: total)) }
        }
        return Trajectory(tSpace: tSpace, systemTrajectory: systemTrajectory, totalTrajectory: totalTrajectory)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: {_, Heff in Heff.add(H) }, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //TODO: Documentation
    @inlinable
    func solve<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        equationType: EquationType,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialState.count == dimension, "The initial state dimension doesn't match the one expected by the hierarchy.")
        var initialTotalState: UniqueVector<Complex<Double>> = .zero(totalDimension)
        for i in 0..<dimension { initialTotalState[i] = initialState[i] }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts.span, equationType: equationType, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
}

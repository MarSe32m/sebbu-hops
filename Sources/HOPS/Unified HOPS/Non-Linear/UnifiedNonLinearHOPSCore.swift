//
//  UnifiedNonLinearHOPSCore.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 11.6.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import BasicContainers

public extension UnifiedHOPSHierarchy {
    /// Solve the non-linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialTotalState: Initial total state, i.e., including the auxiliary states
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - operation: Mapping operation to perform at each simulated time step.
    /// - Returns: The mapped objects.
    @inlinable
    func solveNonLinear<T: ~Copyable, Noise>(
        start: Double = 0.0,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        mapping operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> T
    ) -> UniqueArray<T> where Noise: ComplexNoiseProcess {
        var result: UniqueArray<T> = .init(minimumCapacity: Int((end - start) / stepSize) + 1)
        solveNonLinear(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize) { t, system, total in 
            result.append(operation(t, system, total))
        }
        return result
    }

    /// Solve the non-linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialTotalState: Initial total state, i.e., including the auxiliary states
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - operation: Operation to perform at each simulated time step.
    @inlinable
    func solveNonLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess {
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialTotalState.count == totalDimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
            precondition(noiseShifts.count == G.count, "There must be equal amount of noise shifts as there are exponential terms in the total BCF")
            let initialState: UniqueVector<Complex<Double>> = .init(copying: initialTotalState, count: dimension)
            return withUnsafePointer(to: self) { hierarchy in
                let rhs = NonLinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators)
                let k1 = rhs.zero()
                let k2 = rhs.zero()
                let k3 = rhs.zero()
                let k4 = rhs.zero()
                let temporary = rhs.zero()
                var solver = UniqueRK4Solver(t: start, dt: stepSize, rhs: rhs, k1: k1, k2: k2, k3: k3, k4: k4, temporary: temporary)
                var state = NonLinearHOPSState(totalStateVector: initialTotalState, initialShifts: noiseShifts)
                operation(start, initialState, initialTotalState)
                while solver.t < end {
                    let t = solver.step(y: &state)
                    let systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
                    operation(t, systemState, state.totalStateVector)
                    let _ = systemState.consumeComponents()
                }
                noises = Span()
                customOperators = Span()
            }
        }
    }

    /// Solve the non-linear HOPS equation for this hierarchy structure with QSD jump operators
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation.
    ///   - initialTotalState: Initial total state, i.e., including the auxiliary states.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default values is .none
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - operation: Mapping operation to perform at each simulated time step.
    /// - Returns: The mapped objects.
    @inlinable
    func solveNonLinear<T: ~Copyable, Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        mapping operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> T
    ) -> UniqueArray<T> where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        var result: UniqueArray<T> = .init(minimumCapacity: Int((end - start) / stepSize) + 1)
        solveNonLinear(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize) { t, system, total in 
            result.append(operation(t, system, total))
        }
        return result
    }
    
    /// Solve the non-linear HOPS equation for this hierarchy structure with QSD jump operators
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation.
    ///   - initialTotalState: Initial total state, i.e., including the auxiliary states.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default values is .none
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - operation: Operation to perform at each simulated time step.
    @inlinable
    func solveNonLinear<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        noiseShifts: borrowing Span<Complex<Double>>,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        forEach operation: (Double, borrowing UniqueVector<Complex<Double>>, borrowing UniqueVector<Complex<Double>>) -> Void
    ) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialTotalState.count == totalDimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
            let initialState: UniqueVector<Complex<Double>> = .init(copying: initialTotalState, count: dimension)
            return withUnsafePointer(to: self) { hierarchy in
                let noiseScratch: UnsafeMutableBufferPointer<Complex<Double>> = .allocate(capacity: jumpOperators.count)
                defer { noiseScratch.deallocate() }
                let noiseSpan = noiseScratch.mutableSpan
                let rhs = NonLinearHOPSQSDStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators, jumpOperators: jumpOperators)
                let drift0 = rhs.zero()
                let drift1 = rhs.zero()
                let noise0 = rhs.zero()
                let noise1 = rhs.zero()
                let temporary = rhs.zero()
                var solver = UniqueSRK2Solver(t: start, dt: stepSize, rhs: rhs, drift0: drift0, drift1: drift1, noise0: noise0, noise1: noise1, temporary: temporary, noises: noiseSpan)
                var state = NonLinearHOPSQSDState(totalStateVector: initialTotalState, initialShifts: noiseShifts)
                operation(start, initialState, initialTotalState)
                while solver.t < end {
                    let t = solver.step(y: &state)
                    let systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
                    operation(t, systemState, state.totalStateVector)
                    let _ = systemState.consumeComponents()
                }
                noises = Span()
                customOperators = Span()
                jumpOperators = Span()
            }
        }
    }
}

//
//  UnifiedNonLinearNormalizedHOPSCore.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 24.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import BasicContainers

public extension UnifiedHOPSHierarchy {    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
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
    func solveNonLinearNormalized<T: ~Copyable, Noise>(
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
        solveNonLinearNormalized(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize) { t, system, total in 
            result.append(operation(t, system, total))
        }
        return result
    }

    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
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
    func solveNonLinearNormalized<Noise>(
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
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, equationType: .nonLinearNormalized, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }

    /// Solve the non-linear normalized HOPS equation for this hierarchy structure with QSD jump operators
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
    func solveNonLinearNormalized<T: ~Copyable, Noise, WhiteNoise>(
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
        solveNonLinearNormalized(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize) { t, system, total in 
            result.append(operation(t, system, total))
        }
        return result
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure with QSD jump operators
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
    func solveNonLinearNormalized<Noise, WhiteNoise>(
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
        solve(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, equationType: .nonLinearNormalized, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, forEach: operation)
    }
}

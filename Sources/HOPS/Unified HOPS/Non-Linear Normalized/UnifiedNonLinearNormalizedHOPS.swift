//
//  NonLinearUnifiedHOPS.swift
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
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        return solveNonLinearNormalized(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        return solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        solveNonLinearNormalized(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: Noise,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        return solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: [noises].span, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: Noise,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H = UniqueMatrix<Complex<Double>>(copying: H)
        return solveNonLinearNormalized(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: [noises].span, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: Noise,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        return solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: H, noises: [noises].span, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
        
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default value is .none
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    func solveNonLinearNormalized<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
        precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialTotalStateVector: UniqueVector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialTotalStateVector[i] = initialState[i]
        }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solveNonLinearNormalized(start: start, end: end, initialTotalState: initialTotalStateVector, H: H, noises: noises, noiseShifts: noiseShifts.span, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
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
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        var tSpace: [Double] = []
        var systemTrajectory: [Vector<Complex<Double>>] = []
        var totalTrajectory: [Vector<Complex<Double>>] = []
        solveNonLinearNormalized(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize) { t, system, total in
            tSpace.append(t)
            systemTrajectory.append(.init(copying: system))
            if includeHierarchy { totalTrajectory.append(.init(copying: total)) }
        }
        return Trajectory(tSpace: tSpace, systemTrajectory: systemTrajectory, totalTrajectory: totalTrajectory)
    }
    
    //MARK: QSD + NMQSD versions
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure with QSD jump operators
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default values is .none
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the timestamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(__always)
    func solveNonLinearNormalized<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: Noise,
        shiftType: ShiftType = .none,
        jumpOperators: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: [noises].span, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure with QSD jump operators
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default values is .none
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the timestamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(__always)
    func solveNonLinearNormalized<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        jumpOperators: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure with QSD jump operators
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default values is .none
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the timestamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    func solveNonLinearNormalized<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        jumpOperators: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        return solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: H, noises: noises, shiftType: shiftType, jumpOperators: Span(jumpOperators), customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure with QSD jump operators
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default values is .none
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the timestamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(__always)
    func solveNonLinearNormalized<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinearNormalized(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the non-linear normalized HOPS equation for this hierarchy structure with QSD jump operators
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - shiftType: Shift type for the hierarchy. Default values is .none
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the timestamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    func solveNonLinearNormalized<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
        precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialTotalStateVector: UniqueVector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialTotalStateVector[i] = initialState[i]
        }
        let noiseShifts: [Complex<Double>] = .init(repeating: .zero, count: G.count)
        return solveNonLinearNormalized(start: start, end: end, initialTotalState: initialTotalStateVector, H: H, noises: noises, noiseShifts: noiseShifts.span, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    ///   - includeHierarchy: Flag to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the timestamps, system state vectors, and optionally the full hierarchy states at those time stamps.
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
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        var tSpace: [Double] = []
        var systemTrajectory: [Vector<Complex<Double>>] = []
        var totalTrajectory: [Vector<Complex<Double>>] = []
        solveNonLinearNormalized(start: start, end: end, initialTotalState: initialTotalState, H: H, noises: noises, noiseShifts: noiseShifts, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize) { t, system, total in
            tSpace.append(t)
            systemTrajectory.append(.init(copying: system))
            if includeHierarchy { totalTrajectory.append(.init(copying: total)) }
        }
        return Trajectory(tSpace: tSpace, systemTrajectory: systemTrajectory, totalTrajectory: totalTrajectory)
    }
}

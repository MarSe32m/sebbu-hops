//
//  UnifiedLinearHOPSCorrelationFunctions.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 28.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import BasicContainers

public extension UnifiedHOPSHierarchy {
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        t: Double, A: Matrix<Complex<Double>>,
        s: Double, B: Matrix<Complex<Double>>,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        return solveLinear(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: noises, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H: UniqueMatrix<Complex<Double>> = .init(copying: H)
        return solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        solveLinear(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: noises, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: Noise,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        return solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: [noises].span, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noises: Noise,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H = UniqueMatrix<Complex<Double>>(copying: H)
        return solveLinear(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: [noises].span, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    @inline(always)
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: Noise,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        return solveLinear(start: start, end: end, initialState: initialState, H: H, noises: [noises].span, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
        
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialState: Initial state of the system.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
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
        return solveLinear(start: start, end: end, initialTotalState: initialTotalStateVector, H: H, noises: noises, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy structure
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0.
    ///   - end: End time of the simulation.
    ///   - initialTotalState: Initial total state, i.e., including the auxiliary states.
    ///   - H: The Hamiltonian operator.
    ///   - noises: The environment Gaussian noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy.
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flags to indicate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containing the time stamps, system state vectors, and optionally the full hierarchy states at those time stamps.
    @inlinable
    func solveLinearTwoTimeCorrelationFunction<Noise>(
        start: Double = 0.0,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialTotalState.count == totalDimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial total state (including hierarchy)")
            let initialState: UniqueVector<Complex<Double>> = .init(copying: initialTotalState, count: dimension)
            return withUnsafePointer(to: self) { hierarchy in
                let rhs = LinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, customOperators: customOperators)
                let k1 = rhs.zero()
                let k2 = rhs.zero()
                let k3 = rhs.zero()
                let k4 = rhs.zero()
                let temporary = rhs.zero()
                var solver = UniqueRK4Solver(t: start, dt: stepSize, rhs: rhs, k1: k1, k2: k2, k3: k3, k4: k4, temporary: temporary)
                var state = LinearHOPSState(totalStateVector: initialTotalState)
                var tSpace: [Double] = [0.0]
                var systemTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialState)]
                var totalTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialTotalState)]
                while solver.t < end {
                    let t = solver.step(y: &state)
                    tSpace.append(t)
                    var systemState: Vector<Complex<Double>> = .zero(dimension)
                    state.extractState(into: &systemState)
                    systemTrajectory.append(systemState)
                    if includeHierarchy {
                        totalTrajectory.append(.init(copying: state.totalStateVector))
                    }
                }
                noises = Span()
                customOperators = Span()
                return Trajectory(tSpace: tSpace, systemTrajectory: systemTrajectory, totalTrajectory: totalTrajectory)
            }
        }
    }
    
    //MARK: QSD + NMQSD versions
    
    /// Solve the linear HOPS equation for this hierarchy with QSD terms
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value of 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - noises: The environment Gaussian noise process
    ///   - jumpOperator: The QSD jump operator
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indciate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containign the timestamps, system state vectors, and optionally the full hierarchy states at those timestamps.
    @inlinable
    @inline(__always)
    func solveLinearTwoTimeCorrelationFunction<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: Noise,
        jumpOperators: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: [noises].span, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy with QSD terms
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value of 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - noises: The environment Gaussian noise processes
    ///   - jumpOperator: The QSD jump operator
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indciate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containign the timestamps, system state vectors, and optionally the full hierarchy states at those timestamps.
    @inlinable
    @inline(__always)
    func solveLinearTwoTimeCorrelationFunction<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        jumpOperators: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy with QSD terms
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value of 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - noises: The environment Gaussian noise processes
    ///   - jumpOperator: The QSD jump operator
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indciate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containign the timestamps, system state vectors, and optionally the full hierarchy states at those timestamps.
    @inlinable
    func solveLinearTwoTimeCorrelationFunction<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        jumpOperators: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: H, noises: noises, jumpOperators: Span(jumpOperators), customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy with QSD terms
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value of 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - noises: The environment Gaussian noise processes
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indciate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containign the timestamps, system state vectors, and optionally the full hierarchy states at those timestamps.
    @inlinable
    @inline(__always)
    func solveLinearTwoTimeCorrelationFunction<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy with QSD terms
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value of 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - noises: The environment Gaussian noise processes
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indciate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containign the timestamps, system state vectors, and optionally the full hierarchy states at those timestamps.
    @inlinable
    func solveLinearTwoTimeCorrelationFunction<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
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
        return solveLinear(start: start, end: end, initialTotalState: initialTotalStateVector, H: H, noises: noises, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy with QSD terms
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value of 0.0
    ///   - end: End time of the simulation
    ///   - initialTotalState: Initial total state, i.e., including the auxiliary states
    ///   - H: The Hamiltonian operator
    ///   - noises: The environment Gaussian noise processes
    ///   - jumpOperator: The QSD jump operators
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    ///   - includeHierarchy: Flag to indciate whether to include the whole hierarchy in the resulting trajectory.
    /// - Returns: Trajectory containign the timestamps, system state vectors, and optionally the full hierarchy states at those timestamps.
    @inlinable
    func solveLinearTwoTimeCorrelationFunction<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialTotalState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        jumpOperators: consuming Span<JumpOperator<WhiteNoise>>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialTotalState.count == totalDimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial total state")
            let initialState: UniqueVector<Complex<Double>> = .init(copying: initialTotalState, count: dimension)
            return withUnsafePointer(to: self) { hierarchy in
                let noiseScratch: UnsafeMutableBufferPointer<Complex<Double>> = .allocate(capacity: jumpOperators.count)
                defer { noiseScratch.deallocate() }
                let noiseSpan = noiseScratch.mutableSpan
                let rhs = LinearHOPSQSDStateFunc(hierarchy: hierarchy, H: H, noises: noises, customOperators: customOperators, jumpOperators: jumpOperators)
                let drift0 = rhs.zero()
                let drift1 = rhs.zero()
                let noise0 = rhs.zero()
                let noise1 = rhs.zero()
                let temporary = rhs.zero()
                var solver = UniqueSRK2Solver(t: start, dt: stepSize, rhs: rhs, drift0: drift0, drift1: drift1, noise0: noise0, noise1: noise1, temporary: temporary, noises: noiseSpan)
                var state = LinearHOPSQSDState(totalStateVector: initialTotalState)
                
                var tSpace: [Double] = [0.0]
                var systemTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialState)]
                var totalTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialTotalState)]
                while solver.t < end {
                    let t = solver.step(y: &state)
                    tSpace.append(t)
                    var systemState: Vector<Complex<Double>> = .zero(dimension)
                    state.extractState(into: &systemState)
                    systemTrajectory.append(systemState)
                    if includeHierarchy {
                        totalTrajectory.append(.init(copying: state.totalStateVector))
                    }
                }
                noises = Span()
                customOperators = Span()
                jumpOperators = Span()
                return Trajectory(tSpace: tSpace, systemTrajectory: systemTrajectory, totalTrajectory: totalTrajectory)
            }
        }
    }
}

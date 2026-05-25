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
    @inline(always)
    func solveNonLinear<Noise>(
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
        return solveNonLinear(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
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
    @inline(always)
    func solveNonLinear<Noise>(
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
        return solveNonLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
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
    @inline(always)
    func solveNonLinear<Noise>(
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
        solveNonLinear(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
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
    @inline(always)
    func solveNonLinear<Noise>(
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
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
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
    @inline(always)
    func solveNonLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noise: Noise,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        return solveNonLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: Span(noise), shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
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
    @inline(always)
    func solveNonLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noise: Noise,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H = UniqueMatrix<Complex<Double>>(copying: H)
        return solveNonLinear(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: Span(noise), shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
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
    @inline(always)
    func solveNonLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noise: Noise,
        shiftType: ShiftType = .none,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let noiseSpan = Span(noise)
        return solveNonLinear(start: start, end: end, initialState: initialState, H: H, noises: noiseSpan, shiftType: shiftType, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
        
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
    func solveNonLinear<Noise>(
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
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
            var initialTotalStateVector: UniqueVector<Complex<Double>> = .zero(B.columns)
            for i in 0..<dimension {
                initialTotalStateVector[i] = initialState[i]
            }
            return withUnsafePointer(to: self) { hierarchy in
                let rhs = NonLinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, shiftType: shiftType, customOperators: customOperators)
                let k1 = rhs.zero()
                let k2 = rhs.zero()
                let k3 = rhs.zero()
                let k4 = rhs.zero()
                let temporary = rhs.zero()
                var solver = UniqueRK4Solver(t: start, dt: stepSize, rhs: rhs, k1: k1, k2: k2, k3: k3, k4: k4, temporary: temporary)
                var state = NonLinearHOPSState(totalStateVector: initialTotalStateVector, shiftDimension: G.count)
                var tSpace: [Double] = [0.0]
                var systemTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialState)]
                var totalTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialTotalStateVector)]
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
    
    //MARK: SDE version
    /// Solve the linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoise: The white noise process
    ///   - diffusionOperator: The diffusion operator corresponding to the white noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noise: Noise,
        shiftType: ShiftType = .none,
        jumpOperator: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: Span(noise), shiftType: shiftType, jumpOperator: jumpOperator, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoise: The white noise process
    ///   - diffusionOperator: The diffusion operator corresponding to the white noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        jumpOperator: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, jumpOperator: jumpOperator, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The tmie dependent Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoise: The white noise process
    ///   - diffusionOperator: The diffusion operator corresponding to the white noise process
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding system state vectors
    @inlinable
    func solveNonLinear<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        shiftType: ShiftType = .none,
        jumpOperator: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveNonLinear(start: start, end: end, initialState: initialState, H: H, noises: noises, shiftType: shiftType, jumpOperators: Span(jumpOperator), customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    //MARK: SDE multiple noise version
    /// Solve the linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoises: The white noise processes
    ///   - diffusionOperators: The diffusion operators corresponding to the white noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding system state vectors
    @inlinable
    @inline(__always)
    func solveNonLinear<Noise, WhiteNoise>(
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
        solveNonLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, shiftType: shiftType, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    /// Solve the linear HOPS equation for this hierarchy
    /// - Parameters:
    ///   - start: Start time of the simulation. Default value is 0.0
    ///   - end: End time of the simulation
    ///   - initialState: Initial state of the system
    ///   - H: The tmie dependent Hamiltonian operator
    ///   - z: The environment Gaussian noise process
    ///   - whiteNoises: The white noise processes
    ///   - diffusionOperators: The diffusion operators corresponding to the white noise processes
    ///   - customOperators: Custom linear operators for the diagonal part of the hierarchy
    ///   - stepSize: Simulation step size. Default value is 0.01
    /// - Returns: A tuple containing the time points and the corresponding system state vectors
    @inlinable
    func solveNonLinear<Noise, WhiteNoise>(
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
        withoutActuallyEscaping(H) { H in
            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
            precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
            var initialTotalStateVector: UniqueVector<Complex<Double>> = .zero(B.columns)
            for i in 0..<dimension {
                initialTotalStateVector[i] = initialState[i]
            }
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
                var state = NonLinearHOPSQSDState(totalStateVector: initialTotalStateVector, shiftDimension: G.count)
                var tSpace: [Double] = [0.0]
                var systemTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialState)]
                var totalTrajectory: [Vector<Complex<Double>>] = [.init(copying: initialTotalStateVector)]
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

public extension UnifiedHOPSHierarchy {
    struct NonLinearHOPSStateFunc<Noise: ComplexNoiseProcess>: ~Copyable, ~Escapable, ODERHSFunction {
        @usableFromInline
        internal let hierarchyPointer: UnsafePointer<UnifiedHOPSHierarchy>

        @usableFromInline
        var systemState: UniqueVector<Complex<Double>>
        
        @usableFromInline
        var Heff: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        let H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void
        
        @usableFromInline
        let noises: Span<Noise>
        
        @usableFromInline
        let customOperators: Span<CustomOperator>
        
        @usableFromInline
        var LDaggerExpectationValues: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let WConjugateVector: UniqueVector<Complex<Double>>
        
        @usableFromInline
        var noiseShifts: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @usableFromInline
        let shiftType: ShiftType
        
        @_lifetime(copy noises, copy customOperators)
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
            shiftType: ShiftType,
            customOperators: consuming Span<CustomOperator>
        ) {
            self.dimension = hierarchy.pointee.dimension
            self.totalDimension = hierarchy.pointee.totalDimension
            self.hierarchyPointer = hierarchy
            self.systemState = .zero(hierarchy.pointee.dimension)
            self.Heff = .zeros(rows: hierarchyPointer.pointee.dimension, columns: hierarchyPointer.pointee.dimension)
            self.H = H
            self.noises = noises
            self.customOperators = customOperators
            self.LDaggerExpectationValues = .zero(hierarchy.pointee.LDagger.count)
            self.WConjugateVector = .init(count: hierarchy.pointee.W.count) { buffer in
                for i in hierarchy.pointee.W.indices {
                    buffer[i] = -hierarchy.pointee.W[i].conjugate
                }
            }
            self.noiseShifts = .zero(noises.count)
            self.shiftType = shiftType
        }
        
        @inlinable
        public mutating func evaluate(t: Double, y state: borrowing NonLinearHOPSState, dy result: inout NonLinearHOPSState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            let LDaggerSpan = hierarchyPointer.pointee.LDagger.span
            
            var systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.systemState.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: self.Heff.elements, rows: dimension, columns: dimension)
            var LDaggerExpectations: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.LDaggerExpectationValues.components, count: LDaggerSpan.count)
            var noiseShifts: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.noiseShifts.components, count: noises.count)
            
            for i in 0..<dimension {
                systemState[i] = state.totalStateVector[unchecked: i]
            }
            // Noise shifting
            let normSquared = systemState.normSquared
            for i in 0..<LDaggerExpectations.count {
                LDaggerExpectations[unchecked: i] = systemState.inner(metric: LDaggerSpan[i], systemState) / normSquared
            }
            
            hierarchyPointer.pointee.M.dot(LDaggerExpectations.components, into: result.shiftVector.components)
            for i in 0..<result.shiftVector.count {
                result.shiftVector[unchecked: i] = Relaxed.multiplyAdd(WConjugateVector[unchecked: i], state.shiftVector[unchecked: i], result.shiftVector[unchecked: i])
            }
            for i in hierarchyPointer.pointee.shiftIndices.indices {
                noiseShifts[i] = .zero
                for j in hierarchyPointer.pointee.shiftIndices[i] {
                    noiseShifts[i] = Relaxed.sum(noiseShifts[i], state.shiftVector[j])
                }
            }
            
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            
            for i in LSpan.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i](t).conjugate + noiseShifts[unchecked: i])
                if shiftType == .meanField {
                    Heff.add(LDaggerSpan[unchecked: i], multiplied: -noiseShifts[unchecked: i].conjugate)
                }
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            var kWIndex = 0
            while index < totalDimension {
                Heff.unsafeDot(currentStatePointer, into: resultPointer)
                let kW = kWSpan[unchecked: kWIndex]
                for i in 0..<dimension {
                    resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                }
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
                kWIndex &+= 1
            }
            hierarchyPointer.pointee.B.dot(state.totalStateVector.components, addingInto: result.totalStateVector.components)
            for i in hierarchyPointer.pointee.P.indices {
                hierarchyPointer.pointee.P[i].dot(state.totalStateVector.components, multiplied: LDaggerExpectations[i], addingInto: result.totalStateVector.components)
            }
            if shiftType == .meanField {
                for i in hierarchyPointer.pointee.N.indices {
                    hierarchyPointer.pointee.N[i].dot(state.totalStateVector.components, multiplied: -LDaggerExpectations[i].conjugate, addingInto: result.totalStateVector.components)
                }
            }
            
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
            let _ = LDaggerExpectations.consumeComponents()
            let _ = noiseShifts.consumeComponents()
        }
        
        @inlinable
        public func zero() -> NonLinearHOPSState {
            return .init(dimension: totalDimension, shiftDimension: hierarchyPointer.pointee.G.count)
        }
    }
    
    struct NonLinearHOPSQSDStateFunc<Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess>: ~Copyable, ~Escapable, SDERHSFunction {
        public typealias NoiseType = Complex<Double>
        @usableFromInline
        internal let hierarchyPointer: UnsafePointer<UnifiedHOPSHierarchy>

        @usableFromInline
        var systemState: UniqueVector<Complex<Double>>
        
        @usableFromInline
        var Heff: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        let H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void
        
        @usableFromInline
        let noises: Span<Noise>
        
        @usableFromInline
        let customOperators: Span<CustomOperator>
        
        @usableFromInline
        let jumpOperators: Span<JumpOperator<WhiteNoise>>
        
        @usableFromInline
        var LDaggerExpectationValues: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let WConjugateVector: UniqueVector<Complex<Double>>
        
        @usableFromInline
        var noiseShifts: UniqueVector<Complex<Double>>
        
        @usableFromInline
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @usableFromInline
        let shiftType: ShiftType
        
        @_lifetime(copy noises, copy customOperators, copy jumpOperators)
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
            shiftType: ShiftType,
            customOperators: consuming Span<CustomOperator>,
            jumpOperators: consuming Span<JumpOperator<WhiteNoise>>
        ) {
            self.dimension = hierarchy.pointee.dimension
            self.totalDimension = hierarchy.pointee.totalDimension
            self.hierarchyPointer = hierarchy
            self.systemState = .zero(hierarchyPointer.pointee.dimension)
            self.Heff = .zeros(rows: hierarchyPointer.pointee.dimension, columns: hierarchyPointer.pointee.dimension)
            self.H = H
            self.noises = noises
            self.customOperators = customOperators
            self.jumpOperators = jumpOperators
            self.LDaggerExpectationValues = .zero(hierarchy.pointee.LDagger.count)
            self.WConjugateVector = .init(count: hierarchy.pointee.W.count) { buffer in
                for i in hierarchy.pointee.W.indices {
                    buffer[i] = -hierarchy.pointee.W[i].conjugate
                }
            }
            self.noiseShifts = .zero(noises.count)
            self.shiftType = shiftType
        }
        
        public mutating func drift(t: Double, y state: borrowing NonLinearHOPSQSDState, into result: inout NonLinearHOPSQSDState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            let LDaggerSpan = hierarchyPointer.pointee.LDagger.span
            
            var systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.systemState.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: self.Heff.elements, rows: dimension, columns: dimension)
            var LDaggerExpectations: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.LDaggerExpectationValues.components, count: LDaggerSpan.count)
            var noiseShifts: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.noiseShifts.components, count: noises.count)
            
            for i in 0..<dimension {
                systemState[i] = state.totalStateVector[unchecked: i]
            }
            // Noise shifting
            let normSquared = systemState.normSquared
            for i in 0..<LDaggerExpectations.count {
                LDaggerExpectations[unchecked: i] = systemState.inner(metric: LDaggerSpan[i], systemState) / normSquared
            }
            
            hierarchyPointer.pointee.M.dot(LDaggerExpectations.components, into: result.shiftVector.components)
            for i in 0..<result.shiftVector.count {
                result.shiftVector[unchecked: i] = Relaxed.multiplyAdd(WConjugateVector[unchecked: i], state.shiftVector[unchecked: i], result.shiftVector[unchecked: i])
            }
            for i in hierarchyPointer.pointee.shiftIndices.indices {
                noiseShifts[i] = .zero
                for j in hierarchyPointer.pointee.shiftIndices[i] {
                    noiseShifts[i] = Relaxed.sum(noiseShifts[i], state.shiftVector[j])
                }
            }
            
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            
            for i in LSpan.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i](t).conjugate + noiseShifts[unchecked: i])
                if shiftType == .meanField {
                    Heff.add(LDaggerSpan[unchecked: i], multiplied: -noiseShifts[unchecked: i].conjugate)
                }
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            for i in 0..<jumpOperators.count {
                let gamma = jumpOperators[unchecked: i].rate(t)
                Heff.add(jumpOperators[unchecked: i].LDaggerL, multiplied: -0.5 * gamma)
                // Shift from non-linear QSD
                let jumpOperatorExpectation = systemState.inner(metric: jumpOperators[unchecked: i].jumpOperatorDagger, systemState) / normSquared
                Heff.add(jumpOperators[unchecked: i].jumpOperator, multiplied: gamma * jumpOperatorExpectation)
            }
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            var kWIndex = 0
            while index < totalDimension {
                Heff.unsafeDot(currentStatePointer, into: resultPointer)
                let kW = kWSpan[unchecked: kWIndex]
                for i in 0..<dimension {
                    resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                }
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
                kWIndex &+= 1
            }
            hierarchyPointer.pointee.B.dot(state.totalStateVector.components, addingInto: result.totalStateVector.components)
            for i in hierarchyPointer.pointee.P.indices {
                hierarchyPointer.pointee.P[i].dot(state.totalStateVector.components, multiplied: LDaggerExpectations[i], addingInto: result.totalStateVector.components)
            }
            if shiftType == .meanField {
                for i in hierarchyPointer.pointee.N.indices {
                    hierarchyPointer.pointee.N[i].dot(state.totalStateVector.components, multiplied: -LDaggerExpectations[i].conjugate, addingInto: result.totalStateVector.components)
                }
            }
            
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
            let _ = LDaggerExpectations.consumeComponents()
            let _ = noiseShifts.consumeComponents()
        }
        
        public func diffusion(t: Double, y state: borrowing NonLinearHOPSQSDState, channel: Int, into result: inout NonLinearHOPSQSDState) {
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            while index < totalDimension {
                jumpOperators[unchecked: channel].operate(on: currentStatePointer, into: resultPointer)
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
            }
        }
        
        public func sampleWhiteNoise(t: Double, noises: inout MutableSpan<Complex<Double>>) {
            for i in 0..<jumpOperators.count {
                noises[unchecked: i] = jumpOperators[i].noise(t)
            }
        }

        @inlinable
        public func zero() -> NonLinearHOPSQSDState {
            return .init(dimension: totalDimension, shiftDimension: hierarchyPointer.pointee.G.count)
        }
    }
    
    struct NonLinearHOPSState: ~Copyable, AdaptiveStepODESolverState {
        @inlinable
        public var norm: Double { totalStateVector.norm }
        
        public var totalStateVector: UniqueVector<Complex<Double>>
        public var shiftVector: UniqueVector<Complex<Double>>
        
        @inlinable
        public init(dimension: Int, shiftDimension: Int) {
            self.totalStateVector = .zero(dimension)
            self.shiftVector = .zero(shiftDimension)
        }
        
        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>, shiftDimension: Int) {
            self.totalStateVector = totalStateVector.copy()
            self.shiftVector = .zero(shiftDimension)
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.NonLinearHOPSState) {
            totalStateVector.copyComponents(from: a.totalStateVector)
            shiftVector.copyComponents(from: a.shiftVector)
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.NonLinearHOPSState, multiplied: Double) {
            totalStateVector.copyComponents(from: a.totalStateVector, multiplied: multiplied)
            shiftVector.copyComponents(from: a.shiftVector, multiplied: multiplied)
        }
        
        @inlinable
        public func distance(to: borrowing UnifiedHOPSHierarchy.NonLinearHOPSState) -> Double {
            let totalDistance = totalStateVector.squaredEuclideanDistance(to: to.totalStateVector) + shiftVector.squaredEuclideanDistance(to: to.shiftVector)
            return totalDistance.squareRoot()
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.NonLinearHOPSState, multiplied: Double) {
            totalStateVector.add(a.totalStateVector, multiplied: multiplied)
            shiftVector.add(a.shiftVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.NonLinearHOPSState, adding direction: borrowing UnifiedHOPSHierarchy.NonLinearHOPSState, multipliedBy c: Double) {
            totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
            shiftVector.copyComponents(from: base.shiftVector, adding: direction.shiftVector, multiplied: c)
        }
        
        @inlinable
        public func extractState(into: inout Vector<Complex<Double>>) {
            for i in 0..<into.count {
                into[i] = totalStateVector[unchecked: i]
            }
        }
    }
    
    struct NonLinearHOPSQSDState: ~Copyable, FixedStepSDESolverState {
        public var totalStateVector: UniqueVector<Complex<Double>>
        public var shiftVector: UniqueVector<Complex<Double>>
        
        @inlinable
        public init(dimension: Int, shiftDimension: Int) {
            self.totalStateVector = .zero(dimension)
            self.shiftVector = .zero(shiftDimension)
        }
        
        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>, shiftDimension: Int) {
            self.totalStateVector = totalStateVector.copy()
            self.shiftVector = .zero(shiftDimension)
        }
        
        @inlinable
        public mutating func scale(by: Complex<Double>) {
            totalStateVector.multiply(by: by)
            shiftVector.multiply(by: by)
        }
        
        @inlinable
        public mutating func zero() {
            totalStateVector.zeroComponents()
            shiftVector.zeroComponents()
        }
        
        @inlinable
        public mutating func assign(_ other: borrowing UnifiedHOPSHierarchy.NonLinearHOPSQSDState) {
            totalStateVector.copyComponents(from: other.totalStateVector)
            shiftVector.copyComponents(from: other.shiftVector)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.NonLinearHOPSQSDState, multiplied: Double) {
            totalStateVector.add(a.totalStateVector, multiplied: multiplied)
            shiftVector.add(a.shiftVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.NonLinearHOPSQSDState) {
            totalStateVector.add(a.totalStateVector)
            shiftVector.add(a.shiftVector)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.NonLinearHOPSQSDState, adding direction: borrowing UnifiedHOPSHierarchy.NonLinearHOPSQSDState, multipliedBy c: Double) {
            totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
            shiftVector.copyComponents(from: base.shiftVector, adding: direction.shiftVector, multiplied: c)
        }
        
        @inlinable
        public func extractState(into: inout Vector<Complex<Double>>) {
            for i in 0..<into.count {
                into[i] = totalStateVector[i]
            }
        }
    }
}

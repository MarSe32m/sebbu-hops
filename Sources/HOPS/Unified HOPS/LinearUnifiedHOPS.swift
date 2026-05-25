//
//  LinearUnifiedHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 14.5.2026.
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
    func solveLinear<Noise>(
        start: Double = 0.0,
        end: Double,
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
    func solveLinear<Noise>(
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
    func solveLinear<Noise>(
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
    func solveLinear<Noise>(
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
    func solveLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noise: Noise,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        return solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: [noise].span, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    func solveLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: Vector<Complex<Double>>,
        H: Matrix<Complex<Double>>,
        noise: Noise,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        let H = UniqueMatrix<Complex<Double>>(copying: H)
        return solveLinear(start: start, end: end, initialState: .init(copying: initialState), H: { _, Heff in Heff.add(H) }, noises: [noise].span, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    func solveLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noise: Noise,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess {
        return solveLinear(start: start, end: end, initialState: initialState, H: H, noises: [noise].span, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
        
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
    func solveLinear<Noise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
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
                let rhs = LinearHOPSStateFunc(hierarchy: hierarchy, H: H, noises: noises, customOperators: customOperators)
                let k1 = rhs.zero()
                let k2 = rhs.zero()
                let k3 = rhs.zero()
                let k4 = rhs.zero()
                let temporary = rhs.zero()
                var solver = UniqueRK4Solver(t: start, dt: stepSize, rhs: rhs, k1: k1, k2: k2, k3: k3, k4: k4, temporary: temporary)
                var state = LinearHOPSState(totalStateVector: initialTotalStateVector)
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
    func solveLinear<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noise: Noise,
        jumpOperator: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: [noise].span, jumpOperator: jumpOperator, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    func solveLinear<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: borrowing UniqueMatrix<Complex<Double>>,
        noises: consuming Span<Noise>,
        jumpOperator: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _, Heff in Heff.add(H) }, noises: noises, jumpOperator: jumpOperator, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    func solveLinear<Noise, WhiteNoise>(
        start: Double = 0.0,
        end: Double,
        initialState: borrowing UniqueVector<Complex<Double>>,
        H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
        noises: consuming Span<Noise>,
        jumpOperator: consuming JumpOperator<WhiteNoise>,
        customOperators: consuming Span<CustomOperator> = Span(),
        stepSize: Double = 0.01,
        includeHierarchy: Bool = false
    ) -> Trajectory where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        withUnsafePointer(to: jumpOperator) { jumpOperatorPointer in
            let jumpOperators = Span(_unsafeStart: jumpOperatorPointer, count: 1)
            let result = solveLinear(start: start, end: end, initialState: initialState, H: H, noises: noises, jumpOperators: jumpOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
            noises = Span()
            customOperators = Span()
            return result
        }
        
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
    func solveLinear<Noise, WhiteNoise>(
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
    func solveLinear<Noise, WhiteNoise>(
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
                let rhs = LinearHOPSQSDStateFunc(hierarchy: hierarchy, H: H, noises: noises, customOperators: customOperators, jumpOperators: jumpOperators)
                let drift0 = rhs.zero()
                let drift1 = rhs.zero()
                let noise0 = rhs.zero()
                let noise1 = rhs.zero()
                let temporary = rhs.zero()
                var solver = UniqueSRK2Solver(t: start, dt: stepSize, rhs: rhs, drift0: drift0, drift1: drift1, noise0: noise0, noise1: noise1, temporary: temporary, noises: noiseSpan)
                var state = LinearHOPSQSDState(totalStateVector: initialTotalStateVector)
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

//        withoutActuallyEscaping(H) { H in
//            precondition(noises.count == L.count, "There must be equal amount of noises as there are coupling operators")
//            precondition(initialState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
//            var initialTotalStateVector: Vector<Complex<Double>> = .zero(B.columns)
//            for i in 0..<dimension {
//                initialTotalStateVector[i] = initialState[i]
//            }
//            var systemState = initialState
//            var Heff = H(start)
//            let LSpan = L.span
//            let kWSpan = kWArray.span
//            let noiseSpan = noises.span
//            let jumpOperatorSpan = jumpOperators.span
//            // The solver will deinitialize and deallocate the scratch buffers
//            let diffusionSpace: UnsafeMutableBufferPointer<Vector<Complex<Double>>> = .allocate(capacity: jumpOperators.count)
//            for i in 0..<jumpOperators.count {
//                diffusionSpace.initializeElement(at: i, to: .zero(initialTotalStateVector.count))
//            }
//            let whiteNoiseSpace: UnsafeMutableBufferPointer<Complex<Double>> = .allocate(capacity: jumpOperators.count)
//            var solver = UniqueSRK2Solver(t0: start, initialState: initialTotalStateVector, diffusionSpace: diffusionSpace, noiseSpace: whiteNoiseSpace, dt: stepSize)
//            var tSpace: [Double] = []
//            var systemTrajectory: [Vector<Complex<Double>>] = []
//            var totalTrajectory: [Vector<Complex<Double>>] = []
//            while solver.t < end {
//                solver.step { t, state, result in
//                    for i in 0..<dimension {
//                        systemState[i] = state[i]
//                    }
//                    Heff.copyElements(from: H(t), multiplied: -.i)
//                    for i in LSpan.indices {
//                        Heff._add(LSpan[unchecked: i], multiplied: noiseSpan[unchecked: i](t).conjugate)
//                    }
//                    for customOperator in customOperators {
//                        Heff._add(customOperator(t, systemState))
//                    }
//                    for i in jumpOperatorSpan.indices {
//                        Heff._add(jumpOperatorSpan[unchecked: i].LDaggerL, multiplied: -0.5 * jumpOperatorSpan[unchecked: i].rate(t))
//                    }
//                    result.components.withUnsafeMutableBufferPointer { resultBuffer in
//                        state.components.withUnsafeBufferPointer { currentStateBuffer in
//                            var resultPointer = resultBuffer.baseAddress!
//                            var currentStatePointer = currentStateBuffer.baseAddress!
//                            var index = 0
//                            var kWIndex = 0
//                            while index < resultBuffer.count {
//                                Heff._dot(currentStatePointer, into: resultPointer)
//                                let kW = kWSpan[unchecked: kWIndex]
//                                for i in 0..<dimension {
//                                    resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
//                                }
//                                resultPointer += dimension
//                                currentStatePointer += dimension
//                                index &+= dimension
//                                kWIndex &+= 1
//                            }
//                        }
//                    }
//                    B.dot(state, addingInto: &result)
//                } _: { t, state, result in
//                    state.components.withUnsafeBufferPointer { currentStateBuffer in
//                        for i in result.indices {
//                            result[unchecked: i].components.withUnsafeMutableBufferPointer { resultBuffer in
//                                var resultPointer = resultBuffer.baseAddress!
//                                var currentStatePointer = currentStateBuffer.baseAddress!
//                                var index = 0
//                                while index < resultBuffer.count {
//                                    //jumpOperatorSpan[unchecked: i].operate(on: currentStatePointer, into: resultPointer)
//                                    //jumpOperatorSpan[unchecked: i].jumpOperator._dot(currentStatePointer, into: resultPointer)
//                                    resultPointer += dimension
//                                    currentStatePointer += dimension
//                                    index &+= dimension
//                                }
//                            }
//                        }
//                    }
//                } _: { t, result in
//                    for i in jumpOperatorSpan.indices {
//                        result[unchecked: i] = jumpOperatorSpan[unchecked: i].noise(t)
//                    }
//                } yielding: { t, state in
//                    tSpace.append(t)
//                    var systemState: Vector<Complex<Double>> = .zero(dimension)
//                    systemState.copyComponents(from: state)
//                    systemTrajectory.append(systemState)
//                    if includeHierarchy {
//                        totalTrajectory.append(state)
//                    }
//                }
//            }
//            return Trajectory(tSpace: tSpace, systemTrajectory: systemTrajectory, totalTrajectory: totalTrajectory)
//        }
public extension UnifiedHOPSHierarchy {
    struct LinearHOPSStateFunc<Noise: ComplexNoiseProcess>: ~Copyable, ~Escapable, ODERHSFunction {
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
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @_lifetime(copy noises, copy customOperators)
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
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
        }
        
        @inlinable
        public mutating func evaluate(t: Double, y state: borrowing LinearHOPSState, dy result: inout LinearHOPSState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            var systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: self.systemState.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: self.Heff.elements, rows: dimension, columns: dimension)
            for i in 0..<dimension {
                systemState[i] = state.totalStateVector[unchecked: i]
            }
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            for i in hierarchyPointer.pointee.L.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i](t).conjugate)
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
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
        }
        
        @inlinable
        public func zero() -> LinearHOPSState {
            return .init(dimension: totalDimension)
        }
    }
    
    struct LinearHOPSQSDStateFunc<Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess>: ~Copyable, ~Escapable, SDERHSFunction {
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
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @_lifetime(copy noises, copy customOperators, copy jumpOperators)
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
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
        }
        
        public mutating func drift(t: Double, y state: borrowing LinearHOPSQSDState, into result: inout LinearHOPSQSDState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            var systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: systemState.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: Heff.elements, rows: dimension, columns: dimension)
            for i in 0..<dimension {
                systemState[i] = state.totalStateVector[i]
            }
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            for i in hierarchyPointer.pointee.L.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i](t).conjugate)
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            for i in 0..<jumpOperators.count {
                Heff.add(jumpOperators[i].LDaggerL, multiplied: -0.5 * jumpOperators[i].rate(t))
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
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
        }
        
        public func diffusion(t: Double, y state: borrowing LinearHOPSQSDState, channel: Int, into result: inout LinearHOPSQSDState) {
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
        public func zero() -> LinearHOPSQSDState {
            return .init(dimension: totalDimension)
        }
    }
    
    struct LinearHOPSState: ~Copyable, AdaptiveStepODESolverState {
        @inlinable
        public var norm: Double { totalStateVector.norm }
        
        public var totalStateVector: UniqueVector<Complex<Double>>
        
        @inlinable
        public init(dimension: Int) {
            self.totalStateVector = .zero(dimension)
        }
        
        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>) {
            self.totalStateVector = totalStateVector.copy()
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState) {
            self.totalStateVector.copyComponents(from: a.totalStateVector)
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multiplied: Double) {
            self.totalStateVector.copyComponents(from: a.totalStateVector, multiplied: multiplied)
        }
        
        @inlinable
        public func distance(to: borrowing UnifiedHOPSHierarchy.LinearHOPSState) -> Double {
            self.totalStateVector.euclideanDistance(to: to.totalStateVector)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multiplied: Double) {
            self.totalStateVector.add(a.totalStateVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.LinearHOPSState, adding direction: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multipliedBy c: Double) {
            self.totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
        }
        
        @inlinable
        public func extractState(into: inout Vector<Complex<Double>>) {
            for i in 0..<into.count {
                into[i] = totalStateVector[i]
            }
        }
    }
    
    struct LinearHOPSQSDState: ~Copyable, FixedStepSDESolverState {
        public var totalStateVector: UniqueVector<Complex<Double>>
        
        @inlinable
        public init(dimension: Int) {
            self.totalStateVector = .zero(dimension)
        }
        
        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>) {
            self.totalStateVector = totalStateVector.copy()
        }
        
        @inlinable
        public mutating func scale(by: Complex<Double>) {
            totalStateVector.multiply(by: by)
        }
        
        @inlinable
        public mutating func zero() {
            totalStateVector.zeroComponents()
        }
        
        @inlinable
        public mutating func assign(_ other: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState) {
            totalStateVector.copyComponents(from: other.totalStateVector)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState, multiplied: Double) {
            totalStateVector.add(a.totalStateVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState) {
            totalStateVector.add(a.totalStateVector)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState, adding direction: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState, multipliedBy c: Double) {
            totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
        }
        
        @inlinable
        public func extractState(into: inout Vector<Complex<Double>>) {
            for i in 0..<into.count {
                into[i] = totalStateVector[i]
            }
        }
    }
}

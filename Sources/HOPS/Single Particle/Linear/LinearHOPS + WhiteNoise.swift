//
//  LinearHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

public extension HOPSHierarchy {
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
    func solveLinear<Noise, WhiteNoise>(start: Double = 0.0,
                                        end: Double,
                                        initialState: Vector<Complex<Double>>,
                                        H: Matrix<Complex<Double>>,
                                        z: Noise,
                                        whiteNoise: WhiteNoise,
                                        diffusionOperator: Matrix<Complex<Double>>,
                                        customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                        stepSize: Double = 0.01,
                                        includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, whiteNoise: whiteNoise, diffusionOperator: diffusionOperator, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
    @inlinable
    func linearPropagator<Noise, WhiteNoise>(start: Double,
                                             initialSystemState: Vector<Complex<Double>>,
                                             H: @escaping (Double) -> Matrix<Complex<Double>>,
                                             z: Noise,
                                             whiteNoise: WhiteNoise,
                                             diffusionOperator: Matrix<Complex<Double>>,
                                             customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                             stepSize: Double) -> HOPSPropagator  where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialSystemState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<initialSystemState.count {
            initialStateVector[i] = initialSystemState[i]
        }
        return linearPropagator(start: start, initialHierarchyState: initialStateVector, H: H, z: z, whiteNoise: whiteNoise, diffusionOperator: diffusionOperator, customOperators: customOperators, stepSize: stepSize)
    }
    
    @inlinable
    func linearPropagator<Noise, WhiteNoise>(start: Double,
                                             initialHierarchyState: Vector<Complex<Double>>,
                                             H: @escaping (Double) -> Matrix<Complex<Double>>,
                                             z: Noise,
                                             whiteNoise: WhiteNoise,
                                             diffusionOperator: Matrix<Complex<Double>>,
                                             customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                             stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        var Heff = H(start)
        var systemState: Vector<Complex<Double>> = .zero(dimension)
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialHierarchyState.count)], count: 4)
        let solver = SRK2FixedStep(initialState: [initialHierarchyState], t0: start, dt: stepSize) { t, currentState in
            for i in 0..<dimension {
                systemState[i] = currentState[0][i]
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
            result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                currentState[0].components.withUnsafeBufferPointer { currentStateBuffer in
                    var resultPointer = resultBuffer.baseAddress!
                    var currentStatePointer = currentStateBuffer.baseAddress!
                    var index = 0
                    var kWIndex = 0
                    while index < resultBuffer.count {
                        Heff.dot(currentStatePointer, into: resultPointer)
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
            B.dot(currentState[0], addingInto: &result[0])
            return result
        } g: { t, currentState in
            var result = resultCache.removeFirst()
            defer { resultCache.append(result) }
            result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                currentState[0].components.withUnsafeBufferPointer { currentStateBuffer in
                    var resultPointer = resultBuffer.baseAddress!
                    var currentStatePointer = currentStateBuffer.baseAddress!
                    var index = 0
                    while index < resultBuffer.count {
                        diffusionOperator.dot(currentStatePointer, into: resultPointer)
                        resultPointer += dimension
                        currentStatePointer += dimension
                        index &+= dimension
                    }
                    
                }
            }
            return result
        } w: { t in
            whiteNoise(t)
        }
        return HOPSPropagator(solver)
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
    func solveLinear<Noise, WhiteNoise>(start: Double = 0.0,
                                        end: Double,
                                        initialState: Vector<Complex<Double>>,
                                        H: (Double) -> Matrix<Complex<Double>>,
                                        z: Noise, whiteNoise: WhiteNoise,
                                        diffusionOperator: Matrix<Complex<Double>>,
                                        customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                        stepSize: Double = 0.01,
                                        includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        return withoutActuallyEscaping(H) { H in
            var propagator = linearPropagator(start: start, initialSystemState: initialState, H: H, z: z, whiteNoise: whiteNoise, diffusionOperator: diffusionOperator, customOperators: customOperators, stepSize: stepSize)
            let resultDimension = includeHierarchy ? self.B.columns : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tSpace.capacity)
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
}

//MARK: SDE multiple noise version
public extension HOPSHierarchy {
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
    func solveLinear<Noise, WhiteNoise>(start: Double = 0.0,
                                        end: Double,
                                        initialState: Vector<Complex<Double>>,
                                        H: Matrix<Complex<Double>>,
                                        z: Noise,
                                        whiteNoises: [WhiteNoise],
                                        diffusionOperators: [Matrix<Complex<Double>>],
                                        customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                        stepSize: Double = 0.01,
                                        includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, whiteNoises: whiteNoises, diffusionOperators: diffusionOperators.map { O in {(_: Double, _: Vector<Complex<Double>>) in O }}, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
    }
    
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
    func solveLinear<Noise, WhiteNoise>(start: Double = 0.0,
                                        end: Double,
                                        initialState: Vector<Complex<Double>>,
                                        H: Matrix<Complex<Double>>,
                                        z: Noise,
                                        whiteNoises: [WhiteNoise],
                                        diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                        customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                        stepSize: Double = 0.01,
                                        includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        solveLinear(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, whiteNoises: whiteNoises, diffusionOperators: diffusionOperators, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    func solveLinear<Noise, WhiteNoise>(start: Double = 0.0,
                                        end: Double,
                                        initialState: Vector<Complex<Double>>,
                                        H: (Double) -> Matrix<Complex<Double>>,
                                        z: Noise,
                                        whiteNoises: [WhiteNoise],
                                        diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                        customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [],
                                        stepSize: Double = 0.01,
                                        includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        
        
        return withoutActuallyEscaping(H) { H in
            var propagator = linearPropagator(start: start, initialSystemState: initialState, H: H, z: z, whiteNoises: whiteNoises, diffusionOperators: diffusionOperators, customOperators: customOperators, stepSize: stepSize)
            
            let resultDimension = includeHierarchy ? B.columns : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tSpace.capacity)
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
    func linearPropagator<Noise, WhiteNoise>(start: Double,
                                             initialSystemState: Vector<Complex<Double>>,
                                             H: @escaping (_ t: Double) -> Matrix<Complex<Double>>,
                                             z: Noise,
                                             whiteNoises: [WhiteNoise],
                                             diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                             customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                             stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        precondition(initialSystemState.count == dimension, "The dimension assumed by the hierarchy is not the same as the dimension of the initial state")
        precondition(whiteNoises.count == diffusionOperators.count, "Each white noise process must be associated with a corresponding diffusion operator")
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<initialSystemState.count {
            initialStateVector[i] = initialSystemState[i]
        }
        return linearPropagator(start: start, initialHierarchyState: initialStateVector, H: H, z: z, whiteNoises: whiteNoises, diffusionOperators: diffusionOperators, customOperators: customOperators, stepSize: stepSize)
    }
    
    @inlinable
    func linearPropagator<Noise, WhiteNoise>(start: Double,
                                             initialHierarchyState: Vector<Complex<Double>>,
                                             H: @escaping (_ t: Double) -> Matrix<Complex<Double>>,
                                             z: Noise,
                                             whiteNoises: [WhiteNoise],
                                             diffusionOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                             customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>],
                                             stepSize: Double) -> HOPSPropagator where Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess {
        var systemState: Vector<Complex<Double>> = .zero(dimension)
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialHierarchyState.count)], count: 4)
        // Convert diffusionOperators into the form that the SRK2 solver expects
        let g = diffusionOperators.map { g in
            //TODO: Since the SRK2 solver copies these results into a cache vectors, can we somehow reuse this result vector?
            var result: [Vector<Complex<Double>>] = [.zero(initialHierarchyState.count)]
            var systemState: Vector<Complex<Double>> = .zero(dimension)
            return { (_ t: Double , _ currentState: [Vector<Complex<Double>>]) in
                for i in 0..<dimension {
                    systemState[i] = currentState[0][i]
                }
                let O = g(t, systemState)
                result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                    currentState[0].components.withUnsafeBufferPointer { currentStateBuffer in
                        var resultPointer = resultBuffer.baseAddress!
                        var currentStatePointer = currentStateBuffer.baseAddress!
                        var index = 0
                        while index < resultBuffer.count {
                            O.dot(currentStatePointer, into: resultPointer)
                            resultPointer += dimension
                            currentStatePointer += dimension
                            index &+= dimension
                        }
                        
                    }
                }
                return result
            }
        }
        var Heff = H(start)
        let solver = SRK2FixedStepMultiNoise(initialState: [initialHierarchyState], t0: start, dt: stepSize, f: { t, currentState in
            for i in 0..<dimension {
                systemState[i] = currentState[0][i]
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
            result[0].components.withUnsafeMutableBufferPointer { resultBuffer in
                currentState[0].components.withUnsafeBufferPointer { currentStateBuffer in
                    var resultPointer = resultBuffer.baseAddress!
                    var currentStatePointer = currentStateBuffer.baseAddress!
                    var index = 0
                    var kWIndex = 0
                    while index < resultBuffer.count {
                        Heff.dot(currentStatePointer, into: resultPointer)
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
            B.dot(currentState[0], addingInto: &result[0])
            return result
        }, g: g, w: whiteNoises.map { w in { (_ t: Double) in w(t) } })
        return HOPSPropagator(solver)
    }
}

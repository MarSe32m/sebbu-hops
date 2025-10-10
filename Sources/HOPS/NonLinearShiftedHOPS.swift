//
//  NonLinearShiftedHOPS.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 8.10.2025.
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
    func solveNonLinearShifted<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>], shift: [Complex<Double>]) where Noise: ComplexNoiseProcess {
        solveNonLinearShifted(start: start, end: end, initialState: initialState, H: { _ in H }, z: z, customOperators: customOperators, stepSize: stepSize, includeHierarchy: includeHierarchy)
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
    func solveNonLinearShifted<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: (Double) -> Matrix<Complex<Double>>, z: Noise, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01, includeHierarchy: Bool = false) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>], shift: [Complex<Double>]) where Noise: ComplexNoiseProcess {
        let dimension = initialState.count
        var initialStateVector: Vector<Complex<Double>> = .zero(B.columns)
        for i in 0..<dimension {
            initialStateVector[i] = initialState[i]
        }
        let initialStateVectorForShift: Vector<Complex<Double>> = .zero(G.count)
        let initialStateVectorForHierarchyShift: Vector<Complex<Double>> = .zero(G.count)
        var systemState: Vector<Complex<Double>> = initialState
        let GConjugateVector: Vector<Complex<Double>> = .init(G.map { $0.conjugate })
        let WConjugateVector: Vector<Complex<Double>> = .init(W.map { -$0.conjugate })
        let LDagger = L.conjugateTranspose
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(initialStateVector.count), .zero(G.count), .zero(G.count)], count: 4)
        var _shift: [Complex<Double>] = []
        var creationOperatorPsiCache: [Vector<Complex<Double>>] = .init(repeating: .zero(initialStateVector.count), count: creationOperators.count)
        var annihilationOperatorPsiCache: [Vector<Complex<Double>>] = .init(repeating: .zero(initialStateVector.count), count: annihilationOperators.count)
        var annihilationOperatorPsi0Cache: [Vector<Complex<Double>>] = .init(repeating: .zero(initialStateVector.count), count: annihilationOperators.count)
        
        //TODO: More descriptive names. These are for the computation for the optimal hierarchy (nuHOPS) shift
        var _scratchVector = initialStateVector
        var xpy: Vector<Complex<Double>> = .zero(2 * G.count)
        var D: Matrix<Complex<Double>> = .zeros(rows: 2 * G.count, columns: 2 * G.count)
        return withoutActuallyEscaping(H) { H in
            var Heff = H(start)
            var solver = RK45FixedStep<[Vector<Complex<Double>>]>(initialState: [initialStateVector, initialStateVectorForShift, initialStateVectorForHierarchyShift], t0: start, dt: stepSize) { t, currentStates in
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
                var noiseShift: Complex<Double> = .zero
                
                result[1].components.withUnsafeMutableBufferPointer { result in
                    for i in result.indices {
                        result[i] = Relaxed.multiplyAdd(GConjugateVector[i], LDaggerExpectation, Relaxed.product(WConjugateVector[i], xi[i]))
                        noiseShift = Relaxed.sum(noiseShift, xi[i])
                    }
                }
                let zTilde = z(t).conjugate + noiseShift
                
                // Hierarchy shift
                var hierarchyShift: Complex<Double> = .zero
                let currentShiftVector = currentStates[2]
                
                for i in 0..<currentShiftVector.count {
                    hierarchyShift += currentShiftVector[i]
                }
                
                Heff.zeroElements()
                Heff.add(H(t), multiplied: -.i)
                Heff.add(L, multiplied: zTilde)
                Heff.add(LDagger, multiplied: -hierarchyShift)
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
                
                // Compute hierarchy shift derivative
                for i in G.indices {
                    creationOperators[i].dot(currentState, into: &creationOperatorPsiCache[i])
                    annihilationOperators[i].dot(currentState, into: &annihilationOperatorPsiCache[i])
                    annihilationOperators[i].dot(result[0], into: &annihilationOperatorPsi0Cache[i])
                }
                
                let totalStateNormSquared = currentState.normSquared
                for j in G.indices {
                    let BjExp = currentState.inner(annihilationOperatorPsiCache[j]) / totalStateNormSquared
                    if BjExp.lengthSquared < 1e-2 { continue }
                    var xj = result[0].inner(annihilationOperatorPsiCache[j])
                    xj += currentState.inner(annihilationOperatorPsi0Cache[j])
                    xj -= 2.0 * (BjExp * currentState.inner(result[0])).real
                    xj /= totalStateNormSquared
                    let zj = -W[j] * BjExp * .zero
                    xpy[j] = xj - zj
                    xpy[j + G.count] = xpy[j].conjugate
                    for mu in G.indices {
                        var Amuj = creationOperatorPsiCache[mu].inner(annihilationOperatorPsiCache[j])
                        Amuj -= BjExp * creationOperatorPsiCache[mu].inner(currentState)
                        Amuj /= totalStateNormSquared
                        annihilationOperators[j].dot(creationOperatorPsiCache[mu], into: &_scratchVector)
                        var Cjmu = currentState.inner(_scratchVector)
                        Cjmu -= BjExp * currentState.inner(creationOperatorPsiCache[mu])
                        Cjmu /= totalStateNormSquared
                        
                        D[j, mu] = Cjmu
                        D[j + G.count, mu + G.count] = Cjmu.conjugate
                        D[j, mu + G.count] = Amuj
                        D[j + G.count, mu] = Amuj.conjugate
                    }
                }
                
//                for j in G.indices {
//                    let xj = result[0].inner(annihilationOperatorPsiCache[j])
//                    let yj = currentState.inner(annihilationOperatorPsi0Cache[j])
//                    let zj = -W[j] * currentState.inner(annihilationOperatorPsiCache[j])
//                    xpy[j] = xj + yj + zj
//                    xpy[j + G.count] = xpy[j].conjugate
//                    
//                    for mu in G.indices {
//                        let Amuj = creationOperatorPsiCache[mu].inner(annihilationOperatorPsiCache[j]) / .sqrt(G[mu].conjugate)
//                        annihilationOperators[j].dot(creationOperatorPsiCache[mu], into: &_scratchVector)
//                        let Djmu = currentState.inner(_scratchVector) / .sqrt(G[mu])
//                        D[j, mu] = Djmu
//                        D[j + G.count, mu + G.count] = Djmu.conjugate
//                        
//                        D[j, mu + G.count] = Amuj
//                        D[j + G.count, mu] = Amuj.conjugate
//                        //D[mu, j + G.count] = Amuj
//                        //D[mu + G.count, j] = Amuj.conjugate
//                    }
//                }
                //TODO: Don't allocate mdot every step...
                var n = (try? MatrixOperations.solve(A: D, b: xpy)) ?? .zero(2 * G.count)
                for i in G.indices {
                    // Exact shift
                    var mdot = .sqrt(G[i]) * n[i] - W[i] * currentStates[2][i]
                    result[2][i] = mdot
                    result[0].add(creationOperatorPsiCache[i], multiplied: -n[i])
                    
                    // Mean-field optimized shift
                    //result[2][i] = G[i] * LExpectation - W[i] * currentStates[2][i]
                    //result[0].add(creationOperatorPsiCache[i], multiplied: -.sqrt(G[i]) * LExpectation)
                }
                return result
            }
            let resultDimension = includeHierarchy ? initialStateVector.count : dimension
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            //var trajectory: [Vector<Complex<Double>>] = .init(repeating: .zero(resultDimension), count: tSpace.capacity)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(tSpace.capacity)
            for _ in 0..<trajectory.capacity { trajectory.append(.zero(resultDimension)) }
            var index = 0
            while solver.t < end {
                let (t, state) = solver.step()
                if index >= trajectory.count {
                    trajectory.append(.zero(resultDimension))
                }
                for i in 0..<resultDimension {
                    trajectory[index][i] = state[0][i]
                }
                let shift = state[2].components.reduce(.zero, +)
                _shift.append(shift.conjugate)
                tSpace.append(t)
                index += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory, _shift)
        }
    }
}



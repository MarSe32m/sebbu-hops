//
//  NMQSD.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 11.9.2025.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

public struct NMQSDCalculation: Sendable {
    public let dimension: Int
    public let L: Matrix<Complex<Double>>

    @usableFromInline
    internal let bcf: @Sendable (_ t: Double, _ s: Double) -> Complex<Double>

    @usableFromInline
    internal let G: [Complex<Double>]

    @usableFromInline
    internal let W: [Complex<Double>]

    /// Construct an NMQSDCalculation struct for trajectory computations
    /// - Parameters:
    ///   - dimension: Dimension of the system Hilbert space
    ///   - L: Coupling operator of the system
    public init(dimension: Int, L: Matrix<Complex<Double>>, G: [Complex<Double>], W: [Complex<Double>]) {
        self.dimension = dimension
        self.L = L
        self.bcf = { t, s in 
            var result: Complex<Double> = .zero
            for i in 0..<G.count {
                result = Relaxed.multiplyAdd(G[i], Complex.exp(-(t - s) * W[i]), result)
            }
            return result
        }
        self.G = G
        self.W = W
    }

    @inlinable
    public func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, OBar: (_ t: Double, _ z: Noise) -> Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        var resultCache: Deque<Vector<Complex<Double>>> = .init(repeating: .zero(dimension), count: 4)
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var _LDaggerOBar: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        return withoutActuallyEscaping(OBar) { OBar in 
            var solver = RK45FixedStep(initialState: initialState, t0: start, dt: stepSize) { t, psi in 
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                Heff.copyElements(from: iH)
                Heff.add(L, multiplied: z(t).conjugate)
                LDagger.dot(OBar(t, z), into: &_LDaggerOBar)
                Heff.subtract(_LDaggerOBar)
                Heff.dot(psi, into: &result)
                return result
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 2)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(Int((end - start) / stepSize) + 2)
            for _ in 0..<trajectory.capacity {
                trajectory.append(.zero(dimension))
            }
            var stateIndex = 0
            while solver.t < end {
                let (t, state) = solver.step()
                tSpace.append(t)
                trajectory[stateIndex].copyComponents(from: state)
                stateIndex += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }

    @inlinable
    public func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, z: Noise, OBar: (_ t: Double, _ z: Noise) -> Matrix<Complex<Double>>, customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexNoiseProcess {
        var resultCache: Deque<[Vector<Complex<Double>>]> = .init(repeating: [.zero(dimension), .zero(self.G.count)], count: 4)
        var Heff: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        var _LDaggerOBar: Matrix<Complex<Double>> = .zeros(rows: H.rows, columns: H.columns)
        let LDagger = L.conjugateTranspose
        let iH = -.i * H
        let shiftVectorInitial: Vector<Complex<Double>> = .zero(self.G.count)
        return withoutActuallyEscaping(OBar) { OBar in 
            var solver = RK45FixedStep(initialState: [initialState, shiftVectorInitial], t0: start, dt: stepSize) { t, psi in 
                var result = resultCache.removeFirst()
                defer { resultCache.append(result) }
                let currentState = psi[0]
                let currentShiftVector = psi[1]
                let LDaggerExpectation = currentState.inner(currentState, metric: LDagger) / (currentState.normSquared + 1e-12)
                var noiseShift: Complex<Double> = .zero
                for i in 0..<currentShiftVector.count {
                    noiseShift = Relaxed.sum(noiseShift, currentShiftVector[i])
                    result[1][i] = G[i].conjugate * LDaggerExpectation - W[i].conjugate * currentShiftVector[i]
                }

                Heff.copyElements(from: iH)
                Heff.add(L, multiplied: z(t).conjugate + noiseShift)
                let _OBar = OBar(t, z)
                LDagger.dot(_OBar, into: &_LDaggerOBar)
                Heff.add(_OBar, multiplied: LDaggerExpectation)
                Heff.subtract(_LDaggerOBar)
                Heff.dot(psi[0], into: &result[0])
                return result
            }
            var tSpace: [Double] = []
            tSpace.reserveCapacity(Int((end - start) / stepSize) + 1)
            var trajectory: [Vector<Complex<Double>>] = []
            trajectory.reserveCapacity(Int((end - start) / stepSize) + 1)
            for _ in 0..<trajectory.capacity {
                trajectory.append(.zero(dimension))
            }
            var stateIndex = 0
            while solver.t < end {
                let (t, state) = solver.step()
                tSpace.append(t)
                trajectory[stateIndex].copyComponents(from: state[0])
                stateIndex += 1
            }
            trajectory.removeLast(trajectory.count - tSpace.count)
            return (tSpace, trajectory)
        }
    }

    /// Map a linear NMQSD trajectory to density matrix
    /// - Parameter trajectory: The linear NMQSD trajectory to map to density matrix
    /// - Returns: Array of density matrices
    @inlinable
    @inline(__always)
    public func mapLinearToDensityMatrix(_ trajectory: [Vector<Complex<Double>>]) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate))
        }
        return rho
    }
    
    /// Maps a non-linear NMQSD trajectory to density matrix
    /// - Parameter trajectory: The non-linear NMQSD trajectory to map to density matrix
    /// - Returns: Array of density matrices
    @inlinable
    @inline(__always)
    public func mapNonLinearToDensityMatrix(_ trajectory: [Vector<Complex<Double>>]) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate) / state.normSquared)
        }
        return rho
    }
}

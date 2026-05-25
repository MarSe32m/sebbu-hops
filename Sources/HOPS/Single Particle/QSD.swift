//
//  QSD.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 9.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule

public struct QSDCalculation: Sendable {
    public struct JumpOperator<Noise: ComplexWhiteNoiseProcess> {
        let L: Matrix<Complex<Double>>
        // TODO: Support time-dependent rates
        let rate: Double
        let noise: Noise
        
        public init(L: Matrix<Complex<Double>>, rate: Double, noise: Noise) {
            self.L = L
            self.rate = rate
            self.noise = noise
        }
    }
    
    public init() {
    }

    @inlinable
    public func solveLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, jumpOperators: [JumpOperator<Noise>], customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexWhiteNoiseProcess {
        fatalError()
    }
    
    @inlinable
    public func solveNonLinear<Noise>(start: Double = 0.0, end: Double, initialState: Vector<Complex<Double>>, H: Matrix<Complex<Double>>, jumpOperators: [JumpOperator<Noise>], customOperators: [(_ t: Double, _ state: Vector<Complex<Double>>) -> Matrix<Complex<Double>>] = [], stepSize: Double = 0.01) -> (tSpace: [Double], trajectory: [Vector<Complex<Double>>]) where Noise: ComplexWhiteNoiseProcess {
        fatalError()
    }
    
    /// Maps a QSD trajectory to the corresponding density matrix
    /// - Parameters:
    ///   - trajectory: The QSD trajectory to map to density matrix
    ///   - normalized: Whether the trajectory should be normalized. Default value is false.
    /// - Returns: Array of density matrices
    @inlinable
    public func mapTrajectoryToDensityMatrix(_ trajectory: [Vector<Complex<Double>>], normalize: Bool = false) -> [Matrix<Complex<Double>>] {
        var rho: [Matrix<Complex<Double>>] = []
        rho.reserveCapacity(trajectory.count)
        for state in trajectory {
            rho.append(state.outer(state.conjugate))
            if normalize { rho[rho.count - 1] /= state.normSquared }
        }
        return rho
    }
}

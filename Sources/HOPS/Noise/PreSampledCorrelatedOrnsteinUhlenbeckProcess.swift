//
//  OrnsteinUhlenbeckProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 8.5.2026.
//

import Numerics
import SebbuScience

public struct PreSampledCorrelatedOrnsteinUhlenbeckProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let interpolator: LinearInterpolator<Complex<Double>>
    
    @inlinable
    public init(LC: Matrix<Complex<Double>>, LR: Matrix<Complex<Double>>, F: [Complex<Double>], t: [Double], seed: UInt32 = .random(in: .min ... .max)) {
        var random = NumPyRandom(seed: seed)
        var randomNumbers: [Complex<Double>] = random.nextNormal(count: t.count * F.count, stdev: .sqrt(0.5))
        var xi: Vector<Complex<Double>> = .zero(F.count)
        var x: Vector<Complex<Double>> = .zero(F.count)
        var xNew: Vector<Complex<Double>> = .zero(F.count)
        let one: Vector<Complex<Double>> = .init(.init(repeating: .one, count: F.count))
//        for i in 0..<xi.count { xi[i] = random.nextNormal(stdev: .sqrt(0.5)) }
        for i in 0..<xi.count { xi[i] = randomNumbers.removeLast() }
        LC._dot(xi, into: &x)
        var samples: [Complex<Double>] = [one.inner(x)]
        for _ in 1..<t.count {
            // x_{n+1} = Fx_n + L_R xi_n
//            for i in 0..<xi.count { xi[i] = random.nextNormal(stdev: .sqrt(0.5)) }
            for i in 0..<xi.count { xi[i] = randomNumbers.removeLast() }
            for i in 0..<xNew.count { xNew[i] = x[i] * F[i] }
            LR._dot(xi, addingInto: &xNew)
            samples.append(one.inner(xNew))
            swap(&x, &xNew)
        }
        self.interpolator = LinearInterpolator(x: t, y: samples)
    }
    
    @inlinable
    public init(LC: Matrix<Complex<Double>>, LR: Matrix<Complex<Double>>, F: [Complex<Double>], start: Double, end: Double, dt: Double, seed: UInt32 = .random(in: .min ... .max)) {
        self.init(LC: LC, LR: LR, F: F, t: .linearSpace(start, end, dt), seed: seed)
    }
    
    @inlinable
    public init(r: [Complex<Double>], W: [Complex<Double>], t: [Double], seed: UInt32 = .random(in: .min ... .max)) {
        precondition(r.count == W.count, "The r and W arrays must have the same size.")
        let dt = t[1] - t[0]
        let F: [Complex<Double>] = W.map { .exp(-$0 * dt) }
        var C: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        var R: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        for i in 0..<C.rows {
            for j in 0..<C.columns {
                C[i, j] = r[i] * r[j].conjugate / (W[i] + W[j].conjugate)
                R[i, j] = C[i, j] * (.one - F[i] * F[j].conjugate)
            }
        }
        guard let (D, UVectors) = try? MatrixOperations.diagonalizeHermitian(C) else {
            fatalError("Failed to factorize C")
        }
        guard let (S, VVectors) = try? MatrixOperations.diagonalizeHermitian(R) else {
            fatalError("Failed to factorize R")
        }
        let DMatrix: Matrix<Complex<Double>> = .diagonal(from: D.map { Complex($0.squareRoot()) })
        let U: Matrix<Complex<Double>> = .from(columns: UVectors.map { $0.components })
        let SMatrix: Matrix<Complex<Double>> = .diagonal(from: S.map { Complex($0.squareRoot()) })
        let V: Matrix<Complex<Double>> = .from(columns: VVectors.map { $0.components })
        let LC = U.dot(DMatrix)
        let LR = V.dot(SMatrix)
        self.init(LC: LC, LR: LR, F: F, t: t, seed: seed)
    }
    
    @inlinable
    public init(r: [Complex<Double>], W: [Complex<Double>], start: Double, end: Double, dt: Double, seed: UInt32 = .random(in: .min ... .max)) {
        self.init(r: r, W: W, t: .linearSpace(start, end, dt), seed: seed)
    }
    
    @inlinable
    internal init(_ interpolator: LinearInterpolator<Complex<Double>>) {
        self.interpolator = interpolator
    }
    
    @inlinable
    @inline(always)
    public func sample(_ t: Double) -> Complex<Double> {
        interpolator(t)
    }
    
    @inlinable
    @inline(always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> PreSampledCorrelatedOrnsteinUhlenbeckProcess {
        PreSampledCorrelatedOrnsteinUhlenbeckProcess(LinearInterpolator(x: interpolator.x, y: interpolator.y.map { -$0 }))
    }
}

public struct PreSampledCorrelatedOrnsteinUhlenbeckProcessGenerator: NoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let LC: Matrix<Complex<Double>>
    
    @usableFromInline
    internal let LR: Matrix<Complex<Double>>
    
    @usableFromInline
    internal let F: [Complex<Double>]
    
    @usableFromInline
    internal let t: [Double]
    
    @inlinable
    public init(r: [Complex<Double>], W: [Complex<Double>], t: [Double]) {
        precondition(r.count == W.count, "The r and W arrays must have the same size.")
        let dt = t[1] - t[0]
        let F: [Complex<Double>] = W.map { .exp(-$0 * dt) }
        var C: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        var R: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        for i in 0..<C.rows {
            for j in 0..<C.columns {
                C[i, j] = r[i] * r[j].conjugate / (W[i] + W[j].conjugate)
                R[i, j] = C[i, j] * (.one - F[i] * F[j].conjugate)
            }
        }
        guard let (D, UVectors) = try? MatrixOperations.diagonalizeHermitian(C) else {
            fatalError("Failed to factorize C")
        }
        guard let (S, VVectors) = try? MatrixOperations.diagonalizeHermitian(R) else {
            fatalError("Failed to factorize R")
        }
        let DMatrix: Matrix<Complex<Double>> = .diagonal(from: D.map { Complex($0.squareRoot()) })
        let U: Matrix<Complex<Double>> = .from(columns: UVectors.map { $0.components })
        let SMatrix: Matrix<Complex<Double>> = .diagonal(from: S.map { Complex($0.squareRoot()) })
        let V: Matrix<Complex<Double>> = .from(columns: VVectors.map { $0.components })
        let LC = U.dot(DMatrix)
        let LR = V.dot(SMatrix)
        self.LC = LC
        self.LR = LR
        self.F = F
        self.t = t
    }
    
    @inlinable
    @inline(always)
    public func generate() -> sending PreSampledCorrelatedOrnsteinUhlenbeckProcess {
        PreSampledCorrelatedOrnsteinUhlenbeckProcess(LC: LC, LR: LR, F: F, t: t)
    }
}

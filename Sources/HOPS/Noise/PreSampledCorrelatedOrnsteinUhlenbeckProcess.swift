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
        for i in 0..<xi.count { xi[i] = randomNumbers.removeLast() }
        LC._dot(xi, into: &x)
        var samples: [Complex<Double>] = [one.inner(x)]
        for _ in 1..<t.count {
            // x_{n+1} = Fx_n + L_R xi_n
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
        precondition(t.count >= 2, "Need at least two time points.")
        let dt = t[1] - t[0]
        let (F, LC, LR) = PreSampledCorrelatedOrnsteinUhlenbeckProcessGenerator.constructFLCLR(r: r, W: W, dt: dt)
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
        let (F, LC, LR) = Self.constructFLCLR(r: r, W: W, dt: dt)
        self.F = F
        self.LC = LC
        self.LR = LR
        self.t = t
    }
    
    @inlinable
    public init(r: [Complex<Double>], W: [Complex<Double>], start: Double, end: Double, dt: Double) {
        self.init(r: r, W: W, t: .linearSpace(start, end, dt))
    }
    
    @inlinable
    @inline(always)
    public func generate() -> sending PreSampledCorrelatedOrnsteinUhlenbeckProcess {
        PreSampledCorrelatedOrnsteinUhlenbeckProcess(LC: LC, LR: LR, F: F, t: t)
    }
    
    @inlinable
    internal static func constructFLCLR(r: [Complex<Double>], W: [Complex<Double>], dt: Double) -> (F: [Complex<Double>], LC: Matrix<Complex<Double>>, LR: Matrix<Complex<Double>>) {
        precondition(!r.isEmpty, "Need at least one OU mode.")
        precondition(r.count == W.count, "The r and W arrays must have the same size.")
        precondition(dt > 0.0, "The timestep must be positive.")

        for w in W {
            precondition(w.real > 0.0, "All W must have positive real part.")
            precondition(w.real.isFinite && w.imaginary.isFinite, "W contains non-finite values.")
        }

        for x in r {
            precondition(x.real.isFinite && x.imaginary.isFinite, "r contains non-finite values.")
        }
        let F: [Complex<Double>] = W.map { .exp(-$0 * dt) }
        var C: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        for i in 0..<C.rows {
            for j in 0..<C.columns {
                C[i, j] = r[i] * r[j].conjugate / (W[i] + W[j].conjugate)
            }
        }
        C = MatrixOperations.symmetrizedHermitian(C)
        
        var R: Matrix<Complex<Double>> = .zeros(rows: r.count, columns: r.count)
        for i in 0..<R.rows {
            for j in 0..<R.columns {
                // We do it this way since this is more stable for small time-steps
                let z = (W[i] + W[j].conjugate) * dt
                R[i, j] = r[i] * r[j].conjugate * dt * .phiOneMinusExpMinus(z)
            }
        }
        R = MatrixOperations.symmetrizedHermitian(R)
        let LC = MatrixOperations.positiveSemidefiniteSquareRoot(C)
        let LR = MatrixOperations.positiveSemidefiniteSquareRoot(R)
        return (F, LC, LR)
    }
}


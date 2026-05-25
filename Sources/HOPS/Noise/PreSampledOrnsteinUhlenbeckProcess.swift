//
//  OrnsteinUhlenbeckProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 8.5.2026.
//

import Numerics
import SebbuScience

public struct PreSampledOrnsteinUhlenbeckProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let interpolator: LinearInterpolator<Complex<Double>>
    
    @inlinable
    public init(G: Double, W: Complex<Double>, t: [Double], seed: UInt32 = .random(in: .min ... .max)) {
        var random = NumPyRandom(seed: seed)
        let randomNumbers: [Complex<Double>] = random.nextNormal(count: t.count, stdev: .sqrt(0.5))
        var x = .sqrt(G) * randomNumbers[0]
        var samples: [Complex<Double>] = [x]
        samples.reserveCapacity(t.count)
        let dt = t[1] - t[0]
        let r: Complex<Double> = .exp(-dt * W)
        let B: Double = .sqrt(G * (1 - r.lengthSquared))
        for i in 1..<t.count {
            x = r * x + B * randomNumbers[i]
            samples.append(x)
        }
        self.interpolator = LinearInterpolator(x: t, y: samples)
    }
    
    @inlinable
    public init(G: Double, W: Complex<Double>, start: Double, end: Double, dt: Double, seed: UInt32 = .random(in: .min ... .max)) {
        self.init(G: G, W: W, t: .linearSpace(start, end, dt), seed: seed)
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
    public func antithetic() -> PreSampledOrnsteinUhlenbeckProcess {
        PreSampledOrnsteinUhlenbeckProcess(LinearInterpolator(x: interpolator.x, y: interpolator.y.map { -$0 }))
    }
}

public struct PreSampledOrnsteinUhlenbeckProcessGenerator: NoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let G: Double
    
    @usableFromInline
    internal let W: Complex<Double>
    
    @usableFromInline
    internal let t: [Double]
    
    @inlinable
    public init(G: Double, W: Complex<Double>, t: [Double]) {
        self.G = G
        self.W = W
        self.t = t
    }
    
    @inlinable
    @inline(always)
    public func generate() -> sending PreSampledOrnsteinUhlenbeckProcess {
        PreSampledOrnsteinUhlenbeckProcess(G: G, W: W, t: t)
    }
}

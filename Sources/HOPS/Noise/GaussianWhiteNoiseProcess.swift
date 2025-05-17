//
//  GaussianWhiteNoiseProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics
import SebbuScience

public class GaussianWhiteNoiseProcess: ComplexWhiteNoiseProcess, @unchecked Sendable {
    @usableFromInline
    internal let mean: Double
    @usableFromInline
    internal let deviation: Double
    @usableFromInline
    internal var generator: NumPyRandom
    
    @inlinable
    public init(mean: Double, deviation: Double) {
        self.mean = mean
        self.deviation = deviation.squareRoot()
        self.generator = NumPyRandom()
    }
    
    @inlinable
    @inline(__always)
    public func sample(_ t: Double) -> Complex<Double> {
        generator.nextNormal(mean: mean, stdev: deviation)
    }
    
    @inlinable
    @inline(__always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
}


public struct GaussianWhiteNoiseProcessGenerator: WhiteNoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let mean: Double
    @usableFromInline
    internal let deviation: Double
    
    @inlinable
    public init(mean: Double, deviation: Double) {
        self.mean = mean
        self.deviation = deviation
    }
    
    @inlinable
    @inline(__always)
    public func generate() -> sending GaussianWhiteNoiseProcess {
        GaussianWhiteNoiseProcess(mean: mean, deviation: deviation)
    }
}

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
    internal let mean: (Double) -> Double
    @usableFromInline
    internal let deviation: (Double) -> Double
    @usableFromInline
    internal var generator: NumPyRandom
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: Double) {
        self.mean = { _ in mean }
        self.deviation = { _ in deviation }
        self.generator = NumPyRandom(seed: seed)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @escaping (Double) -> Double, deviation: Double) {
        self.mean = mean
        self.deviation = { _ in deviation }
        self.generator = NumPyRandom(seed: seed)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: @escaping (Double) -> Double) {
        self.mean = { _ in mean }
        self.deviation = deviation
        self.generator = NumPyRandom(seed: seed)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: @escaping (Double) -> Double, deviation: @escaping (Double) -> Double) {
        self.mean = mean
        self.deviation = deviation
        self.generator = NumPyRandom(seed: seed)
    }
    
    @inlinable
    @inline(__always)
    public func sample(_ t: Double) -> Complex<Double> {
        generator.nextNormal(mean: mean(t), stdev: deviation(t))
    }
    
    @inlinable
    @inline(__always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> Self {
        fatalError("Cannot define antithetic for GaussianWhiteNoiseProcess")
    }
}


public struct GaussianWhiteNoiseProcessGenerator: WhiteNoiseProcessGenerator, @unchecked Sendable {
    @usableFromInline
    internal let mean: (Double) -> Double
    @usableFromInline
    internal let deviation: (Double) -> Double
    
    @inlinable
    public init(mean: Double, deviation: Double) {
        self.mean = { _ in mean }
        self.deviation = { _ in deviation }
    }
    
    @inlinable
    public init(mean: @escaping (Double) -> Double, deviation: Double) {
        self.mean = mean
        self.deviation = { _ in deviation }
    }
    
    @inlinable
    public init(mean: Double, deviation: @escaping (Double) -> Double) {
        self.mean = { _ in mean }
        self.deviation = deviation
    }
    
    @inlinable
    public init(mean: @escaping (Double) -> Double, deviation: @escaping (Double) -> Double) {
        self.mean = mean
        self.deviation = deviation
    }
    
    @inlinable
    @inline(__always)
    public func generate() -> sending GaussianWhiteNoiseProcess {
        GaussianWhiteNoiseProcess(mean: mean, deviation: deviation)
    }
}

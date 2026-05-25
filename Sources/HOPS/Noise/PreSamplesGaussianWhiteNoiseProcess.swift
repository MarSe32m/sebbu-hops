//
//  PreSamplesGaussianWhiteNoiseProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics
import SebbuScience

public struct PreSampledGaussianWhiteNoiseProcess: ComplexWhiteNoiseProcess, @unchecked Sendable {
    @usableFromInline
    internal let interpolator: NearestNeighbourInterpolator<Complex<Double>>
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: Double, tSpace: [Double]) {
        self.init(mean: { _ in mean }, deviation: { _ in deviation }, tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: (Double) -> Double, deviation: Double, tSpace: [Double]) {
        self.init(mean: mean, deviation: { _ in deviation }, tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: (Double) -> Double, tSpace: [Double]) {
        self.init(mean: { _ in mean }, deviation: deviation, tSpace: tSpace)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: (Double) -> Double, deviation: (Double) -> Double, tSpace: [Double]) {
        let samples = withoutActuallyEscaping(mean) { mean in
            withoutActuallyEscaping(deviation) { deviation in
                let generatingProcess = GaussianWhiteNoiseProcess(seed: seed, mean: mean, deviation: deviation)
                return tSpace.map { generatingProcess.sample($0)}
            }
        }
        self.interpolator = NearestNeighbourInterpolator(x: tSpace, y: samples)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: { _ in mean }, deviation: { _ in deviation }, start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: (Double) -> Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: mean, deviation: { _ in deviation }, start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: Double, deviation: (Double) -> Double, start: Double, end: Double, step: Double) {
        self.init(seed: seed, mean: { _ in mean }, deviation: deviation, start: start, end: end, step: step)
    }
    
    @inlinable
    public init(seed: UInt32 = .random(in: .min ... .max), mean: (Double) -> Double, deviation: (Double) -> Double, start: Double, end: Double, step: Double) {
        var tSpace: [Double] = []
        tSpace.reserveCapacity(Int((end - start) / step) + 1)
        var t = start
        while t <= end {
            tSpace.append(t)
            t += step
        }
        self.init(mean: mean, deviation: deviation, tSpace: tSpace)
    }
    
    @inlinable
    internal init(interpolator: NearestNeighbourInterpolator<Complex<Double>>) {
        self.interpolator = interpolator
    }
    
    @inlinable
    @inline(__always)
    public func sample(_ t: Double) -> Complex<Double> {
        interpolator(t)
    }
    
    @inlinable
    @inline(__always)
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
    
    @inlinable
    public func antithetic() -> PreSampledGaussianWhiteNoiseProcess {
        let newInterpolator = NearestNeighbourInterpolator(x: interpolator.x, y: interpolator.y.map { -$0 })
        return PreSampledGaussianWhiteNoiseProcess(interpolator: newInterpolator)
    }
}

public struct PreSampledGaussianWhiteNoiseProcessGenerator: WhiteNoiseProcessGenerator, @unchecked Sendable {
    @usableFromInline
    internal let mean: (Double) -> Double
    @usableFromInline
    internal let deviation: (Double) -> Double
    @usableFromInline
    internal let tSpace: [Double]
    
    @inlinable
    public init(mean: Double, deviation: Double, tSpace: [Double]) {
        self.init(mean: { _ in mean }, deviation: { _ in deviation }, tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: @escaping (Double) -> Double, deviation: Double, tSpace: [Double]) {
        self.init(mean: mean, deviation: { _ in deviation }, tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: Double, deviation: @escaping (Double) -> Double, tSpace: [Double]) {
        self.init(mean: { _ in mean }, deviation: deviation, tSpace: tSpace)
    }
    
    @inlinable
    public init(mean: @escaping (Double) -> Double, deviation: @escaping (Double) -> Double, tSpace: [Double]) {
        self.mean = mean
        self.deviation = deviation
        self.tSpace = tSpace
    }
    
    @inlinable
    public init(mean: Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(mean: { _ in mean }, deviation: { _ in deviation }, start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: @escaping (Double) -> Double, deviation: Double, start: Double, end: Double, step: Double) {
        self.init(mean: mean, deviation: { _ in deviation }, start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: Double, deviation: @escaping (Double) -> Double, start: Double, end: Double, step: Double) {
        self.init(mean: { _ in mean }, deviation: deviation, start: start, end: end, step: step)
    }
    
    @inlinable
    public init(mean: @escaping (Double) -> Double, deviation: @escaping (Double) -> Double, start: Double, end: Double, step: Double) {
        var tSpace: [Double] = []
        tSpace.reserveCapacity(Int((end - start) / step) + 1)
        var t = start
        while t <= end {
            tSpace.append(t)
            t += step
        }
        self.init(mean: mean, deviation: deviation, tSpace: tSpace)
    }
    
    @inlinable
    @inline(__always)
    public func generate() -> sending PreSampledGaussianWhiteNoiseProcess {
        PreSampledGaussianWhiteNoiseProcess(mean: mean, deviation: deviation, tSpace: tSpace)
    }
}

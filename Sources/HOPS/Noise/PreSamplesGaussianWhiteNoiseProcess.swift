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
    public init(mean: Double, deviation: Double, tSpace: [Double]) {
        let generatingProcess = GaussianWhiteNoiseProcess(mean: mean, deviation: deviation)
        let samples = tSpace.map { generatingProcess.sample($0) }
        self.interpolator = NearestNeighbourInterpolator(x: tSpace, y: samples)
    }
    
    @inlinable
    public init(mean: Double, deviation: Double, start: Double, end: Double, step: Double) {
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

public struct PreSampledGaussianWhiteNoiseProcessGenerator: WhiteNoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let mean: Double
    @usableFromInline
    internal let deviation: Double
    @usableFromInline
    internal let tSpace: [Double]
    
    @inlinable
    public init(mean: Double, deviation: Double, tSpace: [Double]) {
        self.mean = mean
        self.deviation = deviation
        self.tSpace = tSpace
    }
    
    @inlinable
    public init(mean: Double, deviation: Double, start: Double, end: Double, step: Double) {
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

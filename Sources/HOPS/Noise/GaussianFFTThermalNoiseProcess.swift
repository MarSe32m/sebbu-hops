//
//  GaussianFFTThermalNoiseProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import NumericsExtensions
import SebbuScience

public struct GaussianFFTThermalNoiseProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let spline: CubicHermiteSpline<Complex<Double>>
    
    public var tMax: Double { spline.x.last! }
    
    @inlinable
    public init(temperature: Double, tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01, omegaMax: Double? = nil, spectralDensity: (Double) -> Double) {
        // With zero temperature, this is just zero noise
        if temperature == .zero {
            self.spline = CubicHermiteSpline(x: [0, tMax], y: [0, 0])
            return
        }
        let z = GaussianFFTNoiseProcess(tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax) { omega in
            GaussianFFTThermalNoiseProcess.boseEinstein(omega, temperature) * spectralDensity(omega)
        }
        self.spline = z.spline
    }
    
    @inlinable
    internal init(spline: CubicHermiteSpline<Complex<Double>>) {
        self.spline = spline
    }
    
    @inlinable
    @inline(__always)
    public func sample(_ t: Double) -> ComplexModule.Complex<Double> {
        spline.sample(t)
    }
    
    @inlinable
    public mutating func consumingSample(_ t: Double) -> ComplexModule.Complex<Double> {
        fatalError()
    }
    
    @inlinable
    public func conjugate() -> GaussianFFTThermalNoiseProcess {
        let spline = CubicHermiteSpline(x: spline.x, y: spline.y.map { $0.conjugate })
        return GaussianFFTThermalNoiseProcess(spline: spline)
    }
    
    @inlinable
    @inline(__always)
    internal static func boseEinstein(_ omega: Double, _ temperature: Double) -> Double {
        if omega == .zero || temperature == .zero { return .zero }
        return 1 / (Double.exp(omega/temperature) - 1)
    }
    
}

public struct GaussianFFTThermalNoiseProcessGenerator: NoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let temperature: Double
    @usableFromInline
    internal let tMax: Double
    @usableFromInline
    internal let dtMax: Double
    @usableFromInline
    internal let deltaOmegaMax: Double
    @usableFromInline
    internal let omegaMax: Double?
    @usableFromInline
    internal let spectralDensity: @Sendable (Double) -> Double
    
    @inlinable
    public init(temperature: Double, tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01, omegaMax: Double? = nil, spectralDensity: @Sendable @escaping (_ omega: Double) -> Double) {
        self.temperature = temperature
        self.tMax = tMax
        self.dtMax = dtMax
        self.deltaOmegaMax = deltaOmegaMax
        self.omegaMax = omegaMax
        self.spectralDensity = spectralDensity
    }
    
    @inlinable
    @inline(__always)
    public func generate() -> sending GaussianFFTThermalNoiseProcess {
        GaussianFFTThermalNoiseProcess(temperature: temperature, tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax, spectralDensity: spectralDensity)
    }
}

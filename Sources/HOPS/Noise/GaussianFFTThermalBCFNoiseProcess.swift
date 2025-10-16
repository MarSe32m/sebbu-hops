//
//  GaussianFFTThermalBCFNoiseProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

//
//  GaussianFFTThermalBCFNoiseProcess.swift
//  swift-phd-toivonen
//
//  Created by Sebastian Toivonen on 19.10.2024.
//

import Numerics
import SebbuScience

public struct GaussianFFTThermalBCFNoiseProcess: ComplexNoiseProcess, Sendable {
    @usableFromInline
    internal let spline: CubicHermiteSpline<Complex<Double>>
    
    @inlinable
    public init(temperature: Double, tMax: Double, dtMax: Double = 0.01, deltaOmegaMax: Double = 0.01, omegaMax: Double? = nil, spectralDensity: @escaping (Double) -> Double) {
        // With zero temperature, this is just the same as the typical FFT process
        if temperature == .zero {
            let process = GaussianFFTNoiseProcess(tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax, spectralDensity: spectralDensity)
            self.spline = process.spline
            return
        }
        // Sample the z minus process
        let zMinusProcess = GaussianFFTNoiseProcess(tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax) { omega in
            (GaussianFFTThermalBCFNoiseProcess.boseEinstein(omega, temperature) + 1) * spectralDensity(omega)
        }
        // Sample the z plus process
        let zPlusProcess = GaussianFFTNoiseProcess(tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax) { omega in
            GaussianFFTThermalBCFNoiseProcess.boseEinstein(omega, temperature) * spectralDensity(omega)
        }.conjugate()
        
        let samples = zMinusProcess.spline.x.map { zMinusProcess($0) + zPlusProcess($0) }
        self.spline = CubicHermiteSpline(x: zMinusProcess.spline.x, y: samples)
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
    @inline(__always)
    public mutating func consumingSample(_ t: Double) -> ComplexModule.Complex<Double> {
        fatalError()
    }
    
    @inlinable
    public func conjugate() -> GaussianFFTThermalBCFNoiseProcess {
        let spline = CubicHermiteSpline(x: spline.x, y: spline.y.map { $0.conjugate })
        return GaussianFFTThermalBCFNoiseProcess(spline: spline)
    }
    
    @inlinable
    @inline(__always)
    internal static func boseEinstein(_ omega: Double, _ temperature: Double) -> Double {
        if omega == .zero || temperature == .zero { return .zero }
        return 1 / (Double.exp(omega/temperature) - 1)
    }
    
    @inlinable
    public func antithetic() -> GaussianFFTThermalBCFNoiseProcess {
        let newSpline = CubicHermiteSpline(x: spline.x, y: spline.y.map { -$0 })
        return .init(spline: newSpline)
    }
    
}

public struct GaussianFFTThermalBCFNoiseProcessGenerator: NoiseProcessGenerator, Sendable {
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
    public func generate() -> sending GaussianFFTThermalBCFNoiseProcess {
        GaussianFFTThermalBCFNoiseProcess(temperature: temperature, tMax: tMax, dtMax: dtMax, deltaOmegaMax: deltaOmegaMax, omegaMax: omegaMax, spectralDensity: spectralDensity)
    }
}

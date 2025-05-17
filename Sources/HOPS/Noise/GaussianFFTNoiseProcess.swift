//
//  GaussianFFTNoiseProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics
import SebbuScience

public struct GaussianFFTNoiseProcess: ComplexNoiseProcess, @unchecked Sendable {
    @usableFromInline
    internal var spline: CubicHermiteSpline<Complex<Double>>
    
    public var tMax: Double { spline.x.last! }
    
    @inlinable
    public init(tMax: Double, spectralDensity: (_ omega: Double) -> Double) {
        let deltaOmega = min(.pi / (tMax + 1), 0.001)
        var N = 16
        var dt = 1.0
        while dt > 0.01 {
            N <<= 1
            dt = 2.0 * .pi / (Double(N) * deltaOmega)
        }
        let omegaMax = 2.0 * .pi / dt
        let omegaSpace = [Double].linearSpace(0, omegaMax, N)
        let tSpace = [Double].linearSpace(0, .pi / deltaOmega, N)
        
        //TODO: Maybe implement a more accurate method for random gaussian variables. Maybe Ziggurat algorithm.
        //var generator = SystemRandomNumberGenerator()
        var generator = NumPyRandom()
        let randomCoefficients: [Complex<Double>] = generator.nextNormal(count: N, mean: 0, stdev: .sqrt(0.5))
        var J = [Double](repeating: .zero, count: omegaSpace.count)
        for i in omegaSpace.indices {
            J[i] = spectralDensity(omegaSpace[i])
        }
        let coefficients = omegaSpace.enumerated().map { i, omega in
            return Double.sqrt(deltaOmega * J[i]) * randomCoefficients[i]
        }
        let noise_fft = FFT.fft(coefficients).spectrum
        let _tMax = tSpace.last!
        let _tSpace = [Double].linearSpace(0, _tMax, N/2)
        let splineTSpace = Array(_tSpace[0..<_tSpace.lastIndex(where: {$0 <= tMax})!])
        self.spline = CubicHermiteSpline(x: splineTSpace, y: Array(noise_fft[0..<splineTSpace.count]))
    }
    
    @inlinable
    internal init(spline: CubicHermiteSpline<Complex<Double>>) {
        self.spline = spline
    }
    
    @inlinable
    @inline(__always)
    public func sample(_ t: Double) -> Complex<Double> {
        spline.sample(t)
    }
    
    @inlinable
    public mutating func consumingSample(_ t: Double) -> Complex<Double> {
        fatalError("TODO: Implement")
    }
    
    @inlinable
    public func conjugate() -> GaussianFFTNoiseProcess {
        let spline = CubicHermiteSpline(x: spline.x, y: spline.y.map { $0.conjugate })
        return GaussianFFTNoiseProcess(spline: spline)
    }
}

public struct GaussianFFTNoiseProcessGenerator: NoiseProcessGenerator, Sendable {
    @usableFromInline
    internal let tMax: Double
    @usableFromInline
    internal let spectralDensity: @Sendable (Double) -> Double
    
    @inlinable
    public init(tMax: Double, spectralDensity: @Sendable @escaping (_ omega: Double) -> Double) {
        self.tMax = tMax
        self.spectralDensity = spectralDensity
    }
    
    @inlinable
    @inline(__always)
    public func generate() -> GaussianFFTNoiseProcess {
        GaussianFFTNoiseProcess(tMax: tMax, spectralDensity: spectralDensity)
    }
}

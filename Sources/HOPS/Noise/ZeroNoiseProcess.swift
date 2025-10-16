//
//  ZeroNoiseProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics

public struct ZeroNoiseProcess: Sendable {
    public typealias Element = Complex<Double>
    
    @inlinable
    public init() {}
    
    @inlinable
    public func sample(_ t: Double) -> Complex<Double> {
        .zero
    }
    
    @inlinable
    public func sample(_ t: Double) -> Double {
        .zero
    }
    
    @inlinable
    public func consumingSample(_ t: Double) -> Complex<Double> {
        sample(t)
    }
}

extension ZeroNoiseProcess: ComplexNoiseProcess {
    public func antithetic() -> ZeroNoiseProcess {
        self
    }
}
extension ZeroNoiseProcess: ComplexWhiteNoiseProcess {}

public struct ZeroNoiseProcessGenerator: NoiseProcessGenerator, Sendable {
    
    @inlinable
    public init() {}
    
    @inlinable
    public func generate() -> sending ZeroNoiseProcess {
        ZeroNoiseProcess()
    }
}

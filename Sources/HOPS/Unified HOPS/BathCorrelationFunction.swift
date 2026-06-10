//
//  BathCorrelationFunction.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 12.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import Algorithms

public extension UnifiedHOPSHierarchy {
    struct BathCorrelationFunction: Sendable {
        @usableFromInline
        internal let G: [Complex<Double>]
        
        @usableFromInline
        internal let W: [Complex<Double>]
        
        @usableFromInline
        internal let r: [Complex<Double>]
        
        @usableFromInline
        internal var isZero: Bool { G.isEmpty }
        
        public static var zero: BathCorrelationFunction { BathCorrelationFunction(G: [], W: []) }
        
        public let isPhysical: Bool
        
        @inlinable
        public init(G: [Complex<Double>], W: [Complex<Double>]) {
            precondition(G.count == W.count)
            self.G = G
            self.W = W
            self.r = .init(repeating: .zero, count: G.count)
            self.isPhysical = false
        }
        
        @inlinable
        public init(G: [Complex<Double>], W: [Complex<Double>], r: [Complex<Double>]) {
            precondition(G.count == W.count)
            precondition(G.count == r.count)
            self.G = G
            self.W = W
            self.r = r
            self.isPhysical = true
        }
        
        @inlinable
        public init(tSpace: [Double], terms: Int, fitting bcf: (Double) -> Complex<Double>) {
            self.init(tSpace: tSpace, terms: terms, fitting: tSpace.map { bcf($0) })
        }
        
        @inlinable
        public init(tSpace: [Double], terms: Int, fitting bcf: [Complex<Double>]) {
            precondition(tSpace.count == bcf.count)
            let (G, W) = MatrixPencil.fit(y: bcf, dt: tSpace[1] - tSpace[0], terms: terms)
            self.init(G: G, W: W)
        }
        @inlinable
        public init(tSpace: [Double], terms: Int, physicallyFitting bcf: (Double) -> Complex<Double>) {
            self.init(tSpace: tSpace, terms: terms, physicallyFitting: tSpace.map { bcf($0) })
        }
        
        @inlinable
        public init(tSpace: [Double], terms: Int, physicallyFitting bcf: [Complex<Double>]) {
            precondition(tSpace.count == bcf.count)
            let (G, W, r) = NonLinearFit.fitPhysical(t: tSpace, y: bcf, terms: terms)
            self.init(G: G, W: W, r: r)
        }
        
        @inlinable
        public func callAsFunction(_ t: Double) -> Complex<Double> {
            var result: Complex<Double> = .zero
            let GSpan = G.span
            let WSpan = W.span
            for i in GSpan.indices {
                result += GSpan[unchecked: i] * .exp(-WSpan[unchecked: i] * t)
            }
            return result
        }
        
        @inlinable
        public func callAsFunction(_ t: [Double]) -> [Complex<Double>] {
            t.map { self($0) }
        }
        
        @inlinable
        public func integral(from: Double = .zero, to: Double) -> Complex<Double> {
            let diff = to - from
            var result: Complex<Double> = .zero
            let GSpan = G.span
            let WSpan = W.span
            for i in GSpan.indices {
                result += GSpan[unchecked: i] * WSpan[unchecked: i].reciprocal! * (.one - .exp(-WSpan[unchecked: i] * diff))
            }
            return result
        }
        
        @inlinable
        public func preSampledGenerator(start: Double, end: Double, step: Double) -> PreSampledCorrelatedOrnsteinUhlenbeckProcessGenerator {
            precondition(isPhysical, "The BCF must represent a physical exponential BCF! Meaning that you need to provide the r coefficients")
            return PreSampledCorrelatedOrnsteinUhlenbeckProcessGenerator(r: r, W: W, start: start, end: end, step: step)
        }
        
        @inlinable
        public func generateNoise(start: Double, end: Double, step: Double, seed: UInt32 = .random(in: .min ... .max)) -> PreSampledCorrelatedOrnsteinUhlenbeckProcess {
            precondition(isPhysical, "The BCF must represent a physical exponential BCF! Meaning that you need to provide the r coefficients")
            return PreSampledCorrelatedOrnsteinUhlenbeckProcess(r: r, W: W, start: start, end: end, step: step, seed: seed)
        }
        
        //TODO: Implement
//        @inlinable
//        public func onDemandGenerator(step: Double) -> OnDemandCorrelatedOrnsteinUhlenbeckProcessGenerator {
//            precondition(isPhysical, "The BCF must represent a physical exponential BCF! Meaning that you need to provide the r coefficients")
//            return OnDemandCorrelatedOrnsteinUhlenbeckProcessGenerator(r: r, W: W, dt: dt)
//        }
    }
}

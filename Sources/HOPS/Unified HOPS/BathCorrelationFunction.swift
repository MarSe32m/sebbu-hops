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
        internal var isZero: Bool { G.isEmpty }
        
        public static var zero: BathCorrelationFunction { BathCorrelationFunction(G: [], W: []) }
        
        @inlinable
        public init(G: [Complex<Double>], W: [Complex<Double>]) {
            precondition(G.count == W.count)
            self.G = G
            self.W = W
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
        
        @inline(always)
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
        
        @inline(always)
        @inlinable
        public func callAsFunction(_ t: [Double]) -> [Complex<Double>] {
            t.map { self($0) }
        }
        
        @inline(always)
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
    }
}

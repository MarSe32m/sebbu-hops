//
//  NoiseProcess.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 16.5.2025.
//

import Numerics
import SebbuCollections

public protocol NoiseProcess {
    associatedtype Element
    func callAsFunction(_ t: Double) -> Element
    func sample(_ t: Double) -> Element
    mutating func consumingSample(_ t: Double) -> Element
    func antithetic() -> Self
}

public extension NoiseProcess {
    @inlinable
    @inline(__always)
    func callAsFunction(_ t: Double) -> Element {
        sample(t)
    }
}

public protocol NoiseProcessGenerator {
    associatedtype Process: NoiseProcess & Sendable
    func generate() -> Process
}

public extension NoiseProcessGenerator {
    @inlinable
    @inline(__always)
    func generate(count: Int) -> [Process] {
        (0..<count).map { _ in self.generate() }
    }
}

public extension NoiseProcessGenerator where Self: Sendable {
    @inlinable
    @inline(__always)
    func generateParallel(count: Int) -> [Process] {
        (0..<count).parallelMap { _ in self.generate() }
    }
}

public protocol MultiNoiseProcessGenerator {
    associatedtype Process: NoiseProcess & Sendable
    func generate() -> [Process]
}

public extension MultiNoiseProcessGenerator {
    @inlinable
    @inline(__always)
    func generate(count: Int) -> [[Process]] {
        (0..<count).map { _ in self.generate() }
    }
}

public extension MultiNoiseProcessGenerator where Self: Sendable {
    @inlinable
    @inline(__always)
    func generateParallel(count: Int) -> [[Process]] {
        (0..<count).parallelMap { _ in self.generate() }
    }
}

public protocol RealNoiseProcess: NoiseProcess where Element == Double {}
public protocol ComplexNoiseProcess: NoiseProcess where Element == Complex<Double> {}

public protocol WhiteNoiseProcess: NoiseProcess {}
public protocol RealWhiteNoiseProcess: WhiteNoiseProcess where Element == Double {}
public protocol ComplexWhiteNoiseProcess: WhiteNoiseProcess where Element == Complex<Double> {}

public protocol WhiteNoiseProcessGenerator: NoiseProcessGenerator where Process: WhiteNoiseProcess {}

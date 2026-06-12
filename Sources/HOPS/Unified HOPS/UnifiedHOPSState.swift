//
//  UnifiedHOPSState.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 12.6.2026.
//

import SebbuScience
import Numerics

extension UnifiedHOPSHierarchy {
    @usableFromInline
    internal protocol HOPSStateProtocol: ~Copyable, ~Escapable {
        @inlinable
        var totalStateVector: UniqueVector<Complex<Double>> { get }
        
        @inlinable
        var shiftVector: UniqueVector<Complex<Double>> { get }
    }

    @usableFromInline
    internal struct HOPSState: ~Copyable, AdaptiveStepODESolverState, FixedStepSDESolverState, HOPSStateProtocol {
        @inlinable
        public var norm: Double { totalStateVector.norm }
        
        public var totalStateVector: UniqueVector<Complex<Double>>
        public var shiftVector: UniqueVector<Complex<Double>>
        
        @inlinable
        public init(dimension: Int, shiftDimension: Int) {
            self.totalStateVector = .zero(dimension)
            self.shiftVector = .zero(shiftDimension)
        }
            
        @inlinable
        public init(dimension: Int, initialShifts: borrowing Span<Complex<Double>>) {
            self.totalStateVector = .zero(dimension)
            self.shiftVector = .zero(initialShifts.count)
            for i in 0..<initialShifts.count {
                shiftVector[i] = initialShifts[i]
            }
        }

        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>, shiftDimension: Int) {
            self.totalStateVector = totalStateVector.copy()
            self.shiftVector = .zero(shiftDimension)
        }
        
        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>, initialShifts: borrowing Span<Complex<Double>>) {
            self.totalStateVector = totalStateVector.copy()
            self.shiftVector = .zero(initialShifts.count)
            for i in 0..<initialShifts.count {
                shiftVector[i] = initialShifts[i]
            }
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.HOPSState) {
            totalStateVector.copyComponents(from: a.totalStateVector)
            shiftVector.copyComponents(from: a.shiftVector)
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.HOPSState, multiplied: Double) {
            totalStateVector.copyComponents(from: a.totalStateVector, multiplied: multiplied)
            shiftVector.copyComponents(from: a.shiftVector, multiplied: multiplied)
        }
        
        @inlinable
        public func distance(to: borrowing UnifiedHOPSHierarchy.HOPSState) -> Double {
            let totalDistance = totalStateVector.squaredEuclideanDistance(to: to.totalStateVector) + shiftVector.squaredEuclideanDistance(to: to.shiftVector)
            return totalDistance.squareRoot()
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.HOPSState, multiplied: Double) {
            totalStateVector.add(a.totalStateVector, multiplied: multiplied)
            shiftVector.add(a.shiftVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.HOPSState, adding direction: borrowing UnifiedHOPSHierarchy.HOPSState, multipliedBy c: Double) {
            totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
            shiftVector.copyComponents(from: base.shiftVector, adding: direction.shiftVector, multiplied: c)
        }
        
        @inlinable
        public mutating func zero() {
            totalStateVector.zeroComponents()
            shiftVector.zeroComponents()
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.HOPSState) {
            totalStateVector.add(a.totalStateVector)
            shiftVector.add(a.shiftVector)
        }
        
        @inlinable
        public mutating func scale(by: ComplexModule.Complex<Double>) {
            totalStateVector.multiply(by: by)
            shiftVector.multiply(by: by)
        }
    }
}


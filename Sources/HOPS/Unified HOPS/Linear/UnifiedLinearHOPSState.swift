//
//  LinearHOPSState.swift
//  sebbu-hops
//
//  Created by Sebastian Toivonen on 28.5.2026.
//

import SebbuScience
import Numerics
import SebbuCollections
import DequeModule
import BasicContainers

extension UnifiedHOPSHierarchy {
    @usableFromInline
    internal struct LinearHOPSStateFunc<Noise: ComplexNoiseProcess>: ~Copyable, ~Escapable, ODERHSFunction {
        @usableFromInline
        internal let hierarchyPointer: UnsafePointer<UnifiedHOPSHierarchy>
        
        @usableFromInline
        var Heff: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        let H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void
        
        @usableFromInline
        let noises: Span<Noise>
        
        @usableFromInline
        let customOperators: Span<CustomOperator>
        
        @usableFromInline
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @_lifetime(copy noises, copy customOperators)
        @inlinable
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
            customOperators: consuming Span<CustomOperator>
        ) {
            self.dimension = hierarchy.pointee.dimension
            self.totalDimension = hierarchy.pointee.totalDimension
            self.hierarchyPointer = hierarchy
            self.Heff = .zeros(rows: hierarchyPointer.pointee.dimension, columns: hierarchyPointer.pointee.dimension)
            self.H = H
            self.noises = noises
            self.customOperators = customOperators
        }
        
        @inlinable
        public mutating func evaluate(t: Double, y state: borrowing LinearHOPSState, dy result: inout LinearHOPSState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            var systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: state.totalStateVector.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: self.Heff.elements, rows: dimension, columns: dimension)
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            for i in hierarchyPointer.pointee.L.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i](t).conjugate)
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            var kWIndex = 0
            while index < totalDimension {
                Heff.unsafeDot(currentStatePointer, into: resultPointer)
                let kW = kWSpan[unchecked: kWIndex]
                for i in 0..<dimension {
                    resultPointer[i] = Relaxed.multiplyAdd(kW, currentStatePointer[i], resultPointer[i])
                }
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
                kWIndex &+= 1
            }
            hierarchyPointer.pointee.B.dot(state.totalStateVector.components, addingInto: result.totalStateVector.components)
            let _ = systemState.consumeComponents()
            let _ = Heff.consumeElements()
        }
        
        @inlinable
        public func zero() -> LinearHOPSState {
            return .init(dimension: totalDimension)
        }
    }
    
    @usableFromInline
    internal struct LinearHOPSState: ~Copyable, AdaptiveStepODESolverState {
        @inlinable
        public var norm: Double { totalStateVector.norm }
        
        public var totalStateVector: UniqueVector<Complex<Double>>
        
        @inlinable
        public init(dimension: Int) {
            self.totalStateVector = .zero(dimension)
        }
        
        @inlinable
        public init(totalStateVector: borrowing UniqueVector<Complex<Double>>) {
            self.totalStateVector = totalStateVector.copy()
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState) {
            self.totalStateVector.copyComponents(from: a.totalStateVector)
        }
        
        @inlinable
        public mutating func assign(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multiplied: Double) {
            self.totalStateVector.copyComponents(from: a.totalStateVector, multiplied: multiplied)
        }
        
        @inlinable
        public func distance(to: borrowing UnifiedHOPSHierarchy.LinearHOPSState) -> Double {
            self.totalStateVector.euclideanDistance(to: to.totalStateVector)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multiplied: Double) {
            self.totalStateVector.add(a.totalStateVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.LinearHOPSState, adding direction: borrowing UnifiedHOPSHierarchy.LinearHOPSState, multipliedBy c: Double) {
            self.totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
        }
        
        @inlinable
        public func extractState(into: inout Vector<Complex<Double>>) {
            for i in 0..<into.count {
                into[i] = totalStateVector[i]
            }
        }
    }
}

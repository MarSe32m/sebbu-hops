//
//  LinearHOPSQSDState.swift
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
    internal struct LinearHOPSQSDStateFunc<Noise: ComplexNoiseProcess, WhiteNoise: ComplexWhiteNoiseProcess>: ~Copyable, ~Escapable, SDERHSFunction {
        public typealias NoiseType = Complex<Double>
        @usableFromInline
        internal let hierarchyPointer: UnsafePointer<UnifiedHOPSHierarchy>

        @usableFromInline
        var systemState: UniqueVector<Complex<Double>>
        
        @usableFromInline
        var Heff: UniqueMatrix<Complex<Double>>
        
        @usableFromInline
        let H: (Double, inout UniqueMatrix<Complex<Double>>) -> Void
        
        @usableFromInline
        let noises: Span<Noise>
        
        @usableFromInline
        let customOperators: Span<CustomOperator>
        
        @usableFromInline
        let jumpOperators: Span<JumpOperator<WhiteNoise>>
        
        @usableFromInline
        let dimension: Int
        
        @usableFromInline
        let totalDimension: Int
        
        @_lifetime(copy noises, copy customOperators, copy jumpOperators)
        @inlinable
        public init(
            hierarchy: UnsafePointer<UnifiedHOPSHierarchy>,
            H: @escaping (Double, inout UniqueMatrix<Complex<Double>>) -> Void,
            noises: consuming Span<Noise>,
            customOperators: consuming Span<CustomOperator>,
            jumpOperators: consuming Span<JumpOperator<WhiteNoise>>
        ) {
            self.dimension = hierarchy.pointee.dimension
            self.totalDimension = hierarchy.pointee.totalDimension
            self.hierarchyPointer = hierarchy
            self.systemState = .zero(hierarchyPointer.pointee.dimension)
            self.Heff = .zeros(rows: hierarchyPointer.pointee.dimension, columns: hierarchyPointer.pointee.dimension)
            self.H = H
            self.noises = noises
            self.customOperators = customOperators
            self.jumpOperators = jumpOperators
        }
        
        @inlinable
        public mutating func drift(t: Double, y state: borrowing LinearHOPSQSDState, into result: inout LinearHOPSQSDState) {
            let kWSpan = hierarchyPointer.pointee.kWArray.span
            let LSpan = hierarchyPointer.pointee.L.span
            var systemState: UniqueVector<Complex<Double>> = .init(_unsafeComponents: systemState.components, count: dimension)
            var Heff: UniqueMatrix<Complex<Double>> = .init(_unsafeElements: Heff.elements, rows: dimension, columns: dimension)
            for i in 0..<dimension {
                systemState[i] = state.totalStateVector[i]
            }
            Heff.zeroElements()
            H(t, &Heff)
            Heff.multiply(by: -.i)
            for i in hierarchyPointer.pointee.L.indices {
                Heff.add(LSpan[unchecked: i], multiplied: noises[unchecked: i](t).conjugate)
            }
            for i in 0..<customOperators.count {
                customOperators[unchecked: i](t, systemState, addingTo: &Heff)
            }
            for i in 0..<jumpOperators.count {
                Heff.add(jumpOperators[i].LDaggerL, multiplied: -0.5 * jumpOperators[i].rate(t))
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
        public func diffusion(t: Double, y state: borrowing LinearHOPSQSDState, channel: Int, into result: inout LinearHOPSQSDState) {
            var resultPointer = result.totalStateVector.components
            var currentStatePointer = state.totalStateVector.components
            var index = 0
            while index < totalDimension {
                jumpOperators[unchecked: channel].operate(on: currentStatePointer, into: resultPointer)
                resultPointer += dimension
                currentStatePointer += dimension
                index &+= dimension
            }
        }
        
        @inlinable
        public func sampleWhiteNoise(t: Double, noises: inout MutableSpan<Complex<Double>>) {
            for i in 0..<jumpOperators.count {
                noises[unchecked: i] = jumpOperators[i].noise(t)
            }
        }

        @inlinable
        public func zero() -> LinearHOPSQSDState {
            return .init(dimension: totalDimension)
        }
    }
    
    @usableFromInline
    internal struct LinearHOPSQSDState: ~Copyable, FixedStepSDESolverState {
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
        public mutating func scale(by: Complex<Double>) {
            totalStateVector.multiply(by: by)
        }
        
        @inlinable
        public mutating func zero() {
            totalStateVector.zeroComponents()
        }
        
        @inlinable
        public mutating func assign(_ other: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState) {
            totalStateVector.copyComponents(from: other.totalStateVector)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState, multiplied: Double) {
            totalStateVector.add(a.totalStateVector, multiplied: multiplied)
        }
        
        @inlinable
        public mutating func add(_ a: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState) {
            totalStateVector.add(a.totalStateVector)
        }
        
        @inlinable
        public mutating func assign(_ base: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState, adding direction: borrowing UnifiedHOPSHierarchy.LinearHOPSQSDState, multipliedBy c: Double) {
            totalStateVector.copyComponents(from: base.totalStateVector, adding: direction.totalStateVector, multiplied: c)
        }
        
        @inlinable
        public func extractState(into: inout Vector<Complex<Double>>) {
            for i in 0..<into.count {
                into[i] = totalStateVector[i]
            }
        }
    }
}
